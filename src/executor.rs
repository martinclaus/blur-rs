//! Execution run-times for kernel computations

use std::ops::Range;
use std::slice::from_raw_parts_mut;
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::thread::{self, JoinHandle, Scope};

use crate::data::{Arr2D, Item, Range2D};
use crate::kernel::Kernel;
use rayon::prelude::*;

/// Trait for kernel executors
pub trait Executor {
    /// Apply kernel operation to all valid indices of `res` and populate it with the results.
    fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D);
}

/// Simple serial (single-threaded) executor.
#[derive(Copy, Clone, Debug)]
pub struct SerialExecutor;

impl Executor for SerialExecutor {
    fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D) {
        let shape = res.shape();
        res.iter_mut()
            .zip(shape.iter())
            .for_each(|(d, idx)| *d = K::eval(data, idx))
    }
}

// FIXME: make this a feature
/// Parallel executor build upon rayon's parallel iterator API.
///
/// It relies on the implementation of the ParallelIterator trait
/// of the underlying data structures.
#[derive(Copy, Clone, Debug)]
pub struct RayonIteratorExecutor;

impl Executor for RayonIteratorExecutor {
    fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D) {
        let shape = res.shape();
        // FIXME: use parallel iterator for index
        res.as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out)| {
                *out = K::eval(data, shape.usize_into_index(i));
            });
    }
}

/// Parallel executor build upon Rayon's scoped threads
#[derive(Copy, Clone, Debug)]
pub struct RayonScopeExecutor;

impl Executor for RayonScopeExecutor {
    fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D) {
        let nthreads = rayon::current_num_threads();
        let shape = res.shape();
        let start_idx: Vec<usize> = split_index_range(nthreads, 0..shape.0)
            .map(|r| r.start)
            .collect();
        let res_iter = unsafe { split_mut(res, start_idx.as_slice()) };

        rayon::scope(|s| {
            res_iter.for_each(|(r, res)| {
                s.spawn(move |_| {
                    Range2D(r, 0..shape.1)
                        .zip(res.iter_mut())
                        .for_each(|(idx, r)| {
                            *r = K::eval(data, idx);
                        });
                });
            })
        });
    }
}

/// Parallel executor build upon Rayon's join API
pub struct RayonJoinExecutor;

impl RayonJoinExecutor {
    fn scatter<K: Kernel>(work: &mut [(Range<usize>, &mut [Item])], data: &Arr2D) {
        if work.len() > 1 {
            let (lo, hi) = work.split_at_mut(work.len() / 2);
            rayon::join(
                || Self::scatter::<K>(lo, data),
                || Self::scatter::<K>(hi, data),
            );
        } else {
            // do the actual work
            let (r, res) = &mut work[0];
            Range2D(r.start..r.end, 0..data.shape().1)
                .zip((*res).iter_mut())
                .for_each(|(idx, r)| {
                    *r = K::eval(data, idx);
                });
        }
    }
}

impl Executor for RayonJoinExecutor {
    fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D) {
        let shape = res.shape();
        let start_idx: Vec<usize> = split_index_range(rayon::current_num_threads(), 0..shape.0)
            .map(|r| r.start)
            .collect();
        let mut work_iter = unsafe { split_mut(res, start_idx.as_slice()) };
        Self::scatter::<K>(work_iter.as_mut_slice(), data);
    }
}

/// Returns an iterator of `n_chunks` evenly long sub-ranges
fn split_index_range(n_chunks: usize, range: Range<usize>) -> std::vec::IntoIter<Range<usize>> {
    let mut index_range = vec![];
    let step = range.len() / n_chunks;
    let mut remainder = range.len() % n_chunks;
    let mut start_i0 = vec![range.start];
    let mut next = range.start;
    for _ in 0..n_chunks - 1 {
        if let Some(&last) = start_i0.last() {
            next = last + step;
            if remainder != 0 {
                next += 1;
                remainder -= 1;
            }
            index_range.push(last..next);
            start_i0.push(next);
        }
    }
    index_range.push(next..range.end);
    index_range.into_iter()
}

/// Multi-threaded executor based on std::thread.
///
/// Using scoped threads (Rust >= 1.63) to allow for
/// shared read-only access and Arc + Mutex for shared
/// mutable access.
#[derive(Copy, Clone, Debug)]
pub struct ThreadSharedMutableStateExecutor {
    pub n_threads: usize,
}

impl Executor for ThreadSharedMutableStateExecutor {
    fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D) {
        let shape = res.shape();
        let start_idx: Vec<usize> = split_index_range(self.n_threads, 0..shape.0)
            .map(|r| r.start)
            .collect();
        let res_iter = unsafe { split_mut(res, start_idx.as_slice()) };

        thread::scope(|s| {
            res_iter.for_each(|(r, res)| {
                s.spawn(move || {
                    Range2D(r, 0..shape.1)
                        .zip(res.iter_mut())
                        .for_each(|(idx, r)| {
                            *r = K::eval(data, idx);
                        });
                });
            })
        });
    }
}

/// Build an iterator of ranges over the first dimension of an Arr2D and associated
/// mutable references to the respective data.
///
/// The function will panic if `start_idx` is not monotonic increasing.
///
/// # Safety
/// The referenced slices need to be disjunct. Additional safety requirements are
/// given by [`std::slice::from_raw_parts`].
unsafe fn split_mut<'a>(
    data: &'a mut Arr2D,
    start_idx: &[usize],
) -> std::vec::IntoIter<(Range<usize>, &'a mut [Item])> {
    let shape = data.shape();
    let ptr = data.as_mut_slice().as_mut_ptr();

    let end_point = [&(shape.0)];
    let end_idx = start_idx.iter().skip(1).chain(end_point);

    start_idx
        .iter()
        .zip(end_idx)
        .map(|(&start, &end)| {
            (
                start..end,
                from_raw_parts_mut(
                    ptr.add(start * shape.1),
                    shape.1
                        * end
                            .checked_sub(start)
                            .expect("Negative slice length is not allowed"),
                ),
            )
        })
        .collect::<Vec<_>>()
        .into_iter()
}

/// Multi-threaded Executor based on std:thread and std::sync::mpsc
///
/// Uses scoped threads for shared read-only access but sends the results
/// back to the main thread through a channel. This allows to not use a Mutex
/// but requires an allocation per sended result.
#[derive(Copy, Clone, Debug)]
pub struct ThreadChannelExecutor {
    pub n_threads: usize,
}

impl Executor for ThreadChannelExecutor {
    fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D) {
        let shape = res.shape();

        thread::scope(|s| {
            let index_range =
                split_index_range((2 * self.n_threads).checked_sub(1).unwrap(), 0..shape.0);
            let mut task = SPMDTask::new(
                |j, buff: Option<(usize, Box<[f64]>)>| {
                    let answer = match buff {
                        Some((_, mut buff)) => {
                            (0..shape.1)
                                .zip(buff.iter_mut())
                                .for_each(|(i, a)| *a = K::eval(data, [j, i]));
                            buff
                        }
                        None => (0..shape.1)
                            .map(|i| K::eval(data, [j, i]))
                            .collect::<Vec<Item>>()
                            .into_boxed_slice(),
                    };
                    (j, answer)
                },
                s,
                self.n_threads,
            );

            task.scatter(index_range);
            task.done();
            task.get_result(|(j, buff)| {
                res[j * shape.1..j * shape.1 + buff.len()]
                    .iter_mut()
                    .zip(buff.iter())
                    .for_each(|(r, a)| *r = *a)
            })
            .expect("Task should be done");
        });
    }
}

/// WIP Multi-threaded executor based on std::thread using a fixed pool of worker threads.
///
/// See limitations of `ThreadPool` on why it is not fully usable yet.
#[derive(Debug)]
pub struct ThreadPoolExecutor {
    tp: ThreadPool,
}

impl ThreadPoolExecutor {
    fn split_index_range(nchunks: usize, len: usize) -> Vec<(usize, usize)> {
        let mut index_range = vec![];
        let step = len / nchunks;
        let mut remainder = len % nchunks;
        let mut start_i0 = vec![0];
        let mut next = 0;
        for _ in 0..nchunks - 1 {
            if let Some(last) = start_i0.last() {
                next = last + step;
                if remainder != 0 {
                    next += 1;
                    remainder -= 1;
                }
                index_range.push((*last, next));
                start_i0.push(next);
            }
        }
        index_range.push((next, len));
        index_range
    }

    pub fn new(nthreads: usize) -> Self {
        ThreadPoolExecutor {
            tp: ThreadPool::new(nthreads),
        }
    }
}

impl Executor for ThreadPoolExecutor {
    fn run<'a, K: Kernel>(&self, data: &'a Arr2D, res: &mut Arr2D) {
        let nthreads = self.tp.nthreads();
        let shape = res.shape();
        let nchunks = nthreads * 5;
        let index_range = Self::split_index_range(nchunks, shape.0);

        let task = move |j, buff: Option<(usize, Box<[f64]>)>| {
            let answer = match buff {
                Some((_, mut buff)) => {
                    (0..shape.1)
                        .zip(buff.iter_mut())
                        .for_each(|(i, r)| *r = K::eval(data, [j, i]));
                    buff
                }
                None => (0..shape.1)
                    .map(|i| K::eval(data, [j, i]))
                    .collect::<Vec<Item>>()
                    .into_boxed_slice(),
            };
            (j * shape.1, answer)
        };

        let mut task = self.tp.execute(task);

        task.scatter(index_range.into_iter().map(|(j0, j1)| j0..j1));

        task.done();

        // collect results
        task.get_result(|(i0, a)| {
            res[*i0..i0 + a.len()]
                .iter_mut()
                .zip(a.iter())
                .for_each(|(r, a)| {
                    *r = *a;
                });
        })
        .expect("Should be able to receive results if task is marked done.");
    }
}

type ThreadTask = Box<dyn FnOnce() + Send>;

/// Pool of worker threads.
///
/// Worker threads are spawned on instantiation (`ThreadPool::new(nthreads)`).
/// When `ThreadPool` is dropped, the channels for sending tasks to the threads are
/// disconnected, which causes the threads to finish after completing their last task.
///
/// The tasks send to the thread pool by value have the type `Box<dyn FnOnce() + Send>`.
///
/// The associate method `execute` sets up a task on each worker which can be fed with work
/// through channels by the main thread (i.e. the thread that owns the thread pool).
///
/// # Example
/// Sending a task to a thread pool:
/// ```rust
/// // Crate thread pool with two workers
/// let tp = ThreadPool::new(2);
///
/// // send task to threads
/// tp.task_sender.iter().for_each(|ts| {
///    ts.send(Box::new(|| println!("Hello from thread")))
///        .expect("Should have send task")
/// })
/// ```
#[derive(Debug)]
struct ThreadPool {
    handle: Vec<JoinHandle<()>>,
    task_sender: Vec<SyncSender<ThreadTask>>,
}

impl ThreadPool {
    /// Create a thread pool and spawn `nthreads` of attached worker threads.
    fn new(nthreads: usize) -> Self {
        // spin up worker
        let mut handle = Vec::<JoinHandle<()>>::with_capacity(nthreads);

        let mut task_sender = Vec::<SyncSender<ThreadTask>>::with_capacity(nthreads);
        for _ in 0..nthreads {
            let (sender, receiver) = sync_channel::<ThreadTask>(nthreads);
            task_sender.push(sender);
            handle.push(thread::spawn(|| {
                for task in receiver {
                    task()
                }
            }));
        }

        ThreadPool {
            handle,
            task_sender,
        }
    }

    /// Provide the number of spawned threads.
    fn nthreads(&self) -> usize {
        self.handle.len()
    }

    /// Submit a task of type `Fn(W) -> R` to all workers. The function returns a SPMDTask object
    /// through which work can be send and results can be received.
    /// The task will stop to run when its work sender channel is disconnected, e.g. by dropping
    /// the SPMDTask.
    ///
    /// # Limitations (WIP)
    /// To allow the task to close over a value that has no move semantics
    /// the tasks lifetime is extended to `'static` via the **unsafe** [std::mem::transmute].
    /// However, this requires that the enclosed references lives at least as long as the task is running.
    /// This is not guaranteed at the moment. One way to do it (as done in std::thread) is using a scope
    /// which makes sure that a task is complete before any enclosed reference is dropped.
    fn execute<'a, F, W, R, WI>(&self, task: F) -> SPMDTask<R, WI>
    where
        F: FnOnce(W, Option<R>) -> R + Send + Copy + 'a,
        R: Send + 'static,
        W: Send + 'static,
        WI: Iterator<Item = W> + Send,
    {
        let (result_sender, result_recv) = channel::<(R, Sender<R>)>();
        let mut work_sender = Vec::<Sender<WI>>::with_capacity(self.nthreads());

        {
            self // iterate over slice to not consume task_receiver and thereby disconnect all channels.
                .task_sender
                .iter()
                .for_each(|thread| {
                    // setup channels to send work and receive results
                    let result_sender = result_sender.clone();
                    let (ws, work_recv) = channel::<WI>();
                    work_sender.push(ws);

                    // setup channels to exchange result buffer
                    let (buff_send, buff_recv) = channel::<R>();

                    let b_task = move || {
                        for work_chunk in work_recv {
                            for work in work_chunk {
                                let buff = buff_recv.try_recv().ok();
                                result_sender
                                    .send((task(work, buff), buff_send.clone()))
                                    .expect("Should be able to send result from thread.");
                            }
                        }
                    };

                    // copy closure to heap and change its lifetime to 'static
                    // see https://github.com/crossbeam-rs/crossbeam/blob/70700182c4c92c393fc76209c7acad7af69dca21/crossbeam-utils/src/thread.rs#L445-L446
                    // for a comparison with the implementation in crossbeam
                    let closure: Box<dyn FnOnce() + Send + 'a> = Box::new(b_task);
                    let closure: Box<dyn FnOnce() + Send + 'static> =
                        unsafe { std::mem::transmute(closure) };

                    // send task to thread
                    thread
                        .send(closure)
                        .expect("Should be able to send task to thread");
                });
        }
        SPMDTask {
            work: work_sender,
            result: result_recv,
        }
    }
}

/// A task distributed on a thread pool.
///
/// Work from an iterator can be distributed to the thread pool using `scatter`.
/// Before collecting the results via `get_results`, the task have to be marked by `done`.
///
/// # Example
/// ```rust
/// // start a thread pool
/// let mut task = tp.execute(|i: usize| i + 1);
///
/// // Send work to task
/// task.scatter(0..10);
///
/// // Signal task that no work will be send anymore
/// task.done();
///
/// let mut res = task
///    .get_result()
///    .expect("task is done")
///    .collect::<Vec<usize>>();
/// // we need to sort since the results are unordered
/// res.sort();
///
/// assert!(res.into_iter().zip(1..11).all(|(i1, i2)| i1 == i2));
/// ```
#[derive(Debug)]
struct SPMDTask<R, WI> {
    work: Vec<Sender<WI>>,
    result: Receiver<(R, Sender<R>)>,
}

impl<R, WI> SPMDTask<R, WI> {
    /// create a SPMD task which spawns it own threads within a thread scope
    fn new<'scope, F, W>(f: F, scope: &'scope Scope<'scope, '_>, n_threads: usize) -> Self
    where
        F: FnOnce(W, Option<R>) -> R + Send + Copy + 'scope,
        R: Send + 'scope,
        WI: Iterator<Item = W> + Send + 'scope,
    {
        let n_worker = n_threads
            .checked_sub(1)
            .expect("n_threads should be larger than 0.");

        // setup channel between workers and collector thread
        let (res_sender, res_recv) = channel();

        // Spawn task on worker and collect channels to send work to them
        let work_sender: Vec<Sender<WI>> = (0..n_worker)
            .map(|_| {
                let (buff_send, buff_recv) = channel();
                let (work_sender, work_receiver) = channel();

                let res_sender = res_sender.clone();
                scope.spawn(move || {
                    for work in work_receiver {
                        for work_chunk in work {
                            let buff = buff_recv.try_recv().ok();
                            let res = f(work_chunk, buff);
                            res_sender
                                .send((res, buff_send.clone()))
                                .expect("Result receiver should not be disconnected.")
                        }
                    }
                });
                work_sender
            })
            .collect();

        SPMDTask {
            work: work_sender,
            result: res_recv,
        }
    }

    /// Send work to task
    fn scatter(&self, work: impl Iterator<Item = WI>) {
        work.zip(self.work.iter().cycle()).for_each(|(w, ws)| {
            ws.send(w)
                .expect("Should be able to send work to SPMDTask.")
        })
    }

    /// Spawn thread to collect results and do something with it.
    fn get_result<F>(&self, mut f: F) -> Result<(), &'static str>
    where
        F: FnMut(&R),
    {
        match self.is_done() {
            false => Err("Cannot collect results. SPMDTask not marked as done."),
            true => {
                for (res, buff_sender) in self.result.iter() {
                    f(&res);
                    // Return result buffer to worker. Allow to fail since task may be completed and receiving ends are dropped.
                    buff_sender.send(res).err();
                }
                Ok(())
            }
        }
    }

    /// Signal tasks that no further work will be submitted
    fn done(&mut self) {
        // drop work sender to hang-up channel which stops the task in the thread pool
        self.work.drain(..);
    }

    /// Returns `true` if there can still be send work to the task, `false` otherwise.
    fn is_done(&self) -> bool {
        self.work.is_empty()
    }
}

#[cfg(test)]
mod test {

    use std::slice::Iter;

    use super::{SPMDTask, ThreadPool};

    #[test]
    fn thread_pool_stops_threads_on_drop() {
        let ThreadPool {
            mut handle,
            task_sender,
        } = ThreadPool::new(2);

        // disconnect task sender simulating dropping ThreadPool
        drop(task_sender);

        // wait until all threads have finished
        handle
            .drain(..)
            .for_each(|jh| jh.join().expect("Could not join the associated thread."));
    }

    #[test]
    fn spmd_task_produces_correct_results() {
        let tp = ThreadPool::new(2);
        // let work: Vec<usize> = (0..1000).collect();

        let mut task = tp.execute(|i: usize, _buff: Option<(usize, usize)>| (i, i + 1));
        task.scatter((0..2).map(|i| i * 5..(i + 1) * 5));

        task.done();

        let mut res = [0; 10];
        task.get_result(|(i, r)| {
            res[*i] = *r;
        })
        .expect("task is done");

        assert!(res.into_iter().zip(1..11).all(|(i1, i2)| i1 == i2));
    }

    #[test]
    fn spmd_task_complains_if_task_is_not_done() {
        let tp = ThreadPool::new(2);

        let task: SPMDTask<usize, Iter<usize>> =
            tp.execute(|i: &usize, _buff: Option<usize>| i + 1);

        // will panic if returning an iterator
        let _res = task.get_result(|_| {}).unwrap_err();
    }
}
