use crate::{
    bench::{make_thread_pool, print_arr_sample, run_benchmark},
    executor::{
        RayonExecutor, SerialExecutor, ThreadChannelExecutor, ThreadPoolExecutor,
        ThreadSharedMutableStateExecutor,
    },
};

fn main() {
    // serial execution
    println!("SerialExecutor");
    print_arr_sample(run_benchmark(SerialExecutor {}));

    // parallel execution
    for nthreads in [2, 4, 8, 16] {
        //(2..=16).step_by(2) {
        let executor = RayonExecutor {};
        println!("RayonExecutor ({} threads)", nthreads);
        print_arr_sample(make_thread_pool(nthreads).install(|| run_benchmark(executor)));
    }

    println!("ThreadSharedMutableStateExecutor (8 threads)");
    print_arr_sample(run_benchmark(ThreadSharedMutableStateExecutor {}));

    println!("ThreadChannelExecutor (8 threads)");
    print_arr_sample(run_benchmark(ThreadChannelExecutor {}));

    println!("ThreadPoolExecutor (8 threads)");
    print_arr_sample(run_benchmark(ThreadPoolExecutor::new(8)));
}

mod bench {
    use std::time::Instant;

    use rayon::ThreadPool;

    use crate::{
        data_type::{Arr2D, Shape2D},
        executor::Executor,
        kernel::Blur,
    };

    /// Set up a Rayon thread pool for a given number of threads.
    pub fn make_thread_pool(nthreads: usize) -> ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build()
            .unwrap()
    }

    /// Print slice of the final result
    pub fn print_arr_sample(arr: Arr2D) {
        let i = arr.shape().0 / 2 - arr.shape().1 / 10 - 200;
        print!("Output Sample: ");
        (i..i + 6).for_each(|i| print! {"{:#.2e}, ", arr[[arr.shape().0 / 2, i]]});
        println!();
    }

    /// Mockup data and run a kernel on an executor
    pub fn run_benchmark<E: Executor>(exec: E) -> Arr2D {
        // println!("{}", type_name::<E>());
        let shape = Shape2D(1000, 1000);
        let rep = 100;

        let mut d_in = Arr2D::full(0f64, shape);
        d_in.shape().iter().for_each(|ind| {
            if ((ind[0] as i64) - (shape.0 as i64) / 2).abs() < (shape.1 / 10) as i64
                && ((ind[1] as i64) - (shape.1 as i64) / 2).abs() < (shape.1 / 10) as i64
            {
                d_in[ind] = 1f64;
            } else {
                d_in[ind] = 0f64;
            }
        });

        let mut d_out = Arr2D::full(5f64, shape);

        let now = Instant::now();
        for _ in 0..rep {
            exec.run::<Blur>(&d_in, &mut d_out);
            exec.run::<Blur>(&d_out, &mut d_in);
        }
        println!("Time elapsed: {}", now.elapsed().as_micros() / 2 / rep);
        d_in
    }
}

mod data_type {
    use std::iter::IntoIterator;
    use std::ops::{Index, IndexMut, Range};
    use std::slice::{Iter, IterMut};

    use rayon::prelude::*;

    pub type Ix2 = [usize; 2];
    pub type Item = f64;
    type Range1D = Range<usize>;

    #[derive(Clone, Debug)]

    /// Two-dimensional range of of indices.
    ///
    /// This objects offers an iterator over indices.
    pub struct Range2D(pub Range1D, pub Range1D);

    impl From<Shape2D> for Range2D {
        /// Convert a Shape2D into a Range over all indices.
        #[inline]
        fn from(shape: Shape2D) -> Self {
            Range2D(0..shape.0, 0..shape.1)
        }
    }

    impl Iterator for Range2D {
        type Item = Ix2;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let i = match self.1.next() {
                Some(i) => i,
                None => {
                    self.0.next();
                    self.1.start = 0;
                    self.1.next().ok_or(()).unwrap()
                }
            };
            if self.0.start < self.0.end {
                Some([self.0.start, i])
            } else {
                None
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let hint = self.0.size_hint().0.checked_mul(self.1.size_hint().0);
            (hint.unwrap_or(usize::MAX), hint)
        }
    }

    /// Shape of an Arr2D
    #[derive(Copy, Clone, Debug)]
    pub struct Shape2D(pub usize, pub usize);

    impl Shape2D {
        /// Return an iterator over all valid indices.
        pub fn iter(self) -> Range2D {
            self.into_iter()
        }

        /// Conversion from `Ix2` into linear index.
        ///
        /// 2D indexing is row-major, i.e. last index vary fastest.
        #[inline]
        pub fn index_into_usize(&self, i: Ix2) -> usize {
            self.1 * i[0] + i[1]
        }

        /// Conversion from linear index into `Ix2`.
        ///
        /// 2D indexing is row-major, i.e. last index vary fastest.
        #[inline]
        pub fn usize_into_index(&self, i: usize) -> Ix2 {
            [i / self.1, i % self.1]
        }
    }

    impl IntoIterator for Shape2D {
        type Item = Ix2;
        type IntoIter = Range2D;

        fn into_iter(self) -> Self::IntoIter {
            Range2D::from(self)
        }
    }

    /// 2D Array of a fixed shape
    pub struct Arr2D {
        shape: Shape2D,
        // box slice because of stack size limitations
        data: Box<[Item]>,
    }

    impl Arr2D {
        /// New array filled with constant values.
        pub fn full(item: f64, shape: Shape2D) -> Self {
            Arr2D {
                shape,
                data: vec![item; shape.0 * shape.1].into_boxed_slice(),
            }
        }

        /// Provide the shape of the Array.
        #[inline]
        pub fn shape(&self) -> Shape2D {
            self.shape
        }

        /// Return an iterator over all elements of the array.
        ///
        /// This iterator is a flat iterator and its order is row-major.
        #[inline]
        pub fn iter(&self) -> Iter<Item> {
            self.data.iter()
        }

        /// Return an iterator over mutable references to all elements.
        #[inline]
        pub fn iter_mut(&mut self) -> IterMut<Item> {
            self.data.iter_mut()
        }

        /// Return a Rayon parallel iterator.
        #[inline]
        // FIXME: make this a feature
        pub fn par_iter(&self) -> rayon::slice::Iter<Item> {
            self.data.par_iter()
        }

        /// Return a Rayon parallel iterator over mutable references to all elements.
        #[inline]
        // FIXME: make this a feature
        pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<Item> {
            self.data.par_iter_mut()
        }
    }

    impl Index<Ix2> for Arr2D {
        type Output = Item;
        // row major indexing
        #[inline]
        fn index(&self, index: Ix2) -> &Item {
            &self[self.shape.index_into_usize(index)]
        }
    }

    impl IndexMut<Ix2> for Arr2D {
        // row major indexing
        #[inline]
        fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
            let index = self.shape().index_into_usize(index);
            &mut self[index]
        }
    }

    impl Index<usize> for Arr2D {
        type Output = Item;

        #[inline]
        fn index(&self, index: usize) -> &Self::Output {
            &self.data[index]
        }
    }

    impl IndexMut<usize> for Arr2D {
        #[inline]
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.data[index]
        }
    }

    impl Index<Range1D> for Arr2D {
        type Output = [Item];

        fn index(&self, index: Range1D) -> &Self::Output {
            &self.data[index]
        }
    }

    impl IndexMut<Range1D> for Arr2D {
        fn index_mut(&mut self, index: Range1D) -> &mut Self::Output {
            &mut self.data[index]
        }
    }

    #[cfg(test)]
    mod test {

        use super::{Arr2D, Ix2, Shape2D};

        #[test]
        fn linear_to_tuple_index() {
            let shape = Shape2D(2, 3);
            let res: Vec<Ix2> = shape
                .iter()
                .enumerate()
                .map(|(i, _ind)| shape.usize_into_index(i))
                .collect();
            assert_eq!(res, vec![[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],])
        }

        #[test]
        fn tuple_index_to_linear() {
            let shape = Shape2D(2, 3);
            let res: Vec<usize> = shape
                .iter()
                .map(|ind| shape.index_into_usize(ind))
                .collect();
            let oracle: Vec<usize> = (0..(shape.0 * shape.1)).collect();
            assert_eq!(res, oracle)
        }

        #[test]
        fn range2d_iterates_over_all_indices() {
            let s = Shape2D(2, 3);
            let res: Vec<Ix2> = s.iter().collect();
            assert_eq!(res, vec![[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]);
        }

        #[test]
        fn arr2d_iter_mut_with_index_in_bounds() {
            let shape = Shape2D(500, 100);
            let mut data = Arr2D::full(1f64, shape);
            let data2 = Arr2D::full(2f64, shape);
            shape
                .iter()
                .zip(data.iter_mut())
                .for_each(|(ind, out)| *out = data2[ind]);
            assert!(data.iter().all(|d| *d == 2.0));
        }
    }
}

mod kernel {
    use crate::data_type::{Arr2D, Item, Ix2, Shape2D};

    /// Trait for kernel operations.
    pub trait Kernel {
        /// Evaluate kernel operation for given index idx.
        fn eval(data: &Arr2D, idx: Ix2) -> Item;

        /// Return shape of the kernel
        fn shape() -> Shape2D;

        /// Maps `k_idx`, the index of the kernel item, to an index
        /// of the respective data array, if the kernel shall produce a
        /// value at index `d_idx`.
        ///
        /// # Examples
        /// `Blur` is a centered 3x3 kernel.
        ///
        /// ```
        /// assert_eq!(Blur::map_index([0, 0], [4, 5]), [3, 4]);
        /// assert_eq!(Blur::map_index([2, 1], [4, 5]), [5, 5]);
        /// ```
        fn map_index(k_idx: Ix2, d_idx: Ix2) -> Ix2;
    }

    pub struct Blur;

    impl Blur {
        /// Flags indices as not processable.
        #[inline]
        fn not_process(idx: Ix2, shape: Shape2D) -> bool {
            idx[0] < Self::shape().0
                || idx[0] >= shape.0 - Self::shape().0
                || idx[1] < Self::shape().1
                || idx[1] >= shape.1 - Self::shape().1
        }

        ///
        #[inline]
        const fn kernel() -> [Item; 9] {
            const KERNEL_GAUSS: [Item; 9] = [
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                4.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
            ];
            KERNEL_GAUSS
        }
    }

    impl Kernel for Blur {
        fn eval(data: &Arr2D, idx: Ix2) -> Item {
            if Self::not_process(idx, data.shape()) {
                return data[idx];
            }

            let mut sum: Item = 0.0;

            // convolve with kernel
            Self::shape()
                .iter()
                .zip(Self::kernel().iter())
                .for_each(|(k_idx, elem)| {
                    sum += elem * data[Self::map_index(k_idx, idx)];
                });

            sum
        }

        #[inline]
        fn shape() -> Shape2D {
            const KERNEL_GAUSS_SHAPE: Shape2D = Shape2D(3, 3);
            KERNEL_GAUSS_SHAPE
        }

        #[inline]
        fn map_index(k_ind: Ix2, data_ind: Ix2) -> Ix2 {
            [data_ind[0] + k_ind[0] - 1, data_ind[1] + k_ind[1] - 1]
        }
    }

    #[cfg(test)]
    mod test {
        use crate::kernel::{Blur, Kernel};

        // FIXME: is doc test, remove when crate is turned into library
        #[test]
        fn map_index_does_produce_correct_results() {
            assert_eq!(Blur::map_index([0, 0], [4, 5]), [3, 4]);
            assert_eq!(Blur::map_index([2, 1], [4, 5]), [5, 5]);
        }
    }
}

mod executor {
    use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};

    use crate::data_type::{Arr2D, Item, Range2D};
    use crate::kernel::Kernel;
    use rayon::prelude::*;

    /// Trait for kernel executors
    pub trait Executor {
        /// Apply kernel operation to all valid indices of `res` and populate it with the results.
        fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D);
    }

    /// Simple serial (single-threaded) executor.
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
    /// Parallel executor build upon rayon library.
    ///
    /// It relies on the implementation of the ParallelIterator trait
    /// of the underlying data structures.
    pub struct RayonExecutor;

    impl Executor for RayonExecutor {
        fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D) {
            let shape = res.shape();
            // FIXME: use parallel iterator for index
            res.par_iter_mut().enumerate().for_each(|(i, out)| {
                *out = K::eval(data, shape.usize_into_index(i));
            });
        }
    }

    /// Multi-threaded executor based on std::thread.
    ///
    /// Using scoped threads (Rust >= 1.63) to allow for
    /// shared read-only access and Arc + Mutex for shared
    /// mutable access.
    pub struct ThreadSharedMutableStateExecutor;

    impl ThreadSharedMutableStateExecutor {
        fn split_index_range(nthreads: usize, len: usize) -> Vec<(usize, usize)> {
            let mut index_range = vec![];
            let step = len / nthreads;
            let mut remainder = len % nthreads;
            let mut start_i0 = vec![0];
            let mut next = 0;
            for _ in 0..nthreads - 1 {
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
    }

    impl Executor for ThreadSharedMutableStateExecutor {
        fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D) {
            let nthreads = 8;
            let shape = res.shape();
            let index_range = Self::split_index_range(nthreads, shape.0);

            let res = Arc::new(Mutex::new(res));

            thread::scope(|s| {
                for (i0, i1) in index_range {
                    let res = res.clone();
                    s.spawn(move || {
                        // pre-allocated for better performance
                        let mut answer = Vec::<Item>::with_capacity((i1 - i0) * shape.1);
                        Range2D(i0..i1, 0..shape.1).for_each(|idx| answer.push(K::eval(data, idx)));

                        let mut res = res.lock().unwrap();
                        res[i0 * shape.1..i1 * shape.1 - 1]
                            .iter_mut()
                            .zip(answer)
                            .for_each(|(r, a)| *r = a);
                    });
                }
            });
        }
    }

    /// Multi-threaded Executor based on std:thread and std::sync::mpsc
    ///
    /// Uses scoped threads for shared read-only access but sends the results
    /// back to the main thread through a channel. This allows to not use a Mutex
    /// but requires an allocation per sended result.
    pub struct ThreadChannelExecutor;

    impl ThreadChannelExecutor {
        fn split_index_range(nthreads: usize, len: usize) -> Vec<(usize, usize)> {
            let mut index_range = vec![];
            let step = len / nthreads;
            let mut remainder = len % nthreads;
            let mut start_i0 = vec![0];
            let mut next = 0;
            for _ in 0..nthreads - 1 {
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
    }

    impl Executor for ThreadChannelExecutor {
        fn run<K: Kernel>(&self, data: &Arr2D, res: &mut Arr2D) {
            let nthreads = 8;
            let shape = res.shape();
            let index_range = Self::split_index_range(nthreads, shape.0);

            thread::scope(|s| {
                let (tx, rv) = channel();
                for (i0, i1) in index_range {
                    let tx = tx.clone();
                    let (buff_send, buff_recv) = channel();

                    s.spawn(move || {
                        for j in i0..i1 {
                            let mut answer = match buff_recv.try_recv() {
                                Ok(buff) => buff,
                                Err(_) => (0..shape.1)
                                    .map(|_| 0.0)
                                    .collect::<Vec<Item>>()
                                    .into_boxed_slice(),
                            };
                            (0..shape.1)
                                .zip(answer.iter_mut())
                                .for_each(|(i, a)| *a = K::eval(data, [j, i]));
                            tx.send((j, answer, buff_send.clone())).unwrap();
                        }
                    });
                }

                // Need to drop tx to finally hang-up the result channel so that we an iterate over it.
                drop(tx);

                for (j, answer, return_buff) in rv {
                    res[j * shape.1..(j + 1) * shape.1]
                        .iter_mut()
                        .zip(answer.iter())
                        .for_each(|(r, a)| {
                            *r = *a;
                        });
                    return_buff.send(answer).err();
                }
            });
        }
    }

    /// WIP Multi-threaded executor based on std::thread using a fixed pool of worker threads.
    ///
    /// See limitations of `ThreadPool` on why it is not fully usable yet.
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

            let mut task = self.tp.execute(move |(i0, i1)| {
                let answer = Range2D(i0..i1, 0..shape.1)
                    .map(|idx| K::eval(data, idx))
                    .collect::<Vec<Item>>()
                    .into_boxed_slice();
                (i0 * shape.1, answer)
            });

            task.scatter(index_range.into_iter());

            task.done();

            // collect results
            for (i0, a) in task.get_result().expect("Task should be marked `done`") {
                res[i0..i0 + a.len()]
                    .iter_mut()
                    .zip(a.iter())
                    .for_each(|(r, a)| {
                        *r = *a;
                    });
            }
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
        fn execute<'a, F, W, R>(&self, task: F) -> SPMDTask<W, R>
        where
            F: FnOnce(W) -> R + Send + Copy + 'a,
            R: Send + 'static,
            W: Send + 'static,
        {
            let (result_sender, result_recv) = channel::<R>();
            let mut work_sender = Vec::<Sender<W>>::with_capacity(self.nthreads());

            {
                self // iterate over slice to not consume task_receiver and thereby disconnect all channels.
                    .task_sender
                    .iter()
                    .for_each(|thread| {
                        // setup channels to send work and receive results
                        let result_sender = result_sender.clone();
                        let (ws, work_recv) = channel::<W>();
                        work_sender.push(ws);

                        let b_task = move || {
                            for work in work_recv {
                                result_sender
                                    .send(task(work))
                                    .expect("Should be able to send result from thread.");
                            }
                        };

                        // copy closure to heap and change its lifetime to 'static
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
    struct SPMDTask<W, R> {
        work: Vec<Sender<W>>,
        result: Receiver<R>,
    }

    impl<W, R> SPMDTask<W, R> {
        /// Send work to task
        fn scatter(&self, work: impl Iterator<Item = W>) {
            work.zip(self.work.iter().cycle()).for_each(|(w, ws)| {
                ws.send(w)
                    .expect("Should be able to send work to SPMDTask.")
            })
        }

        /// Return a blocking iterator of unordered result values, but only if there cannot be
        /// any more work send to the task to prevent a deadlock.
        fn get_result(&self) -> Result<std::sync::mpsc::Iter<R>, &'static str> {
            match self.is_done() {
                true => Ok(self.result.iter()),
                false => Err("Cannot collect results. SPMDTask not marked as done."),
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

        use super::ThreadPool;

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

            let mut task = tp.execute(|i: usize| i + 1);
            task.scatter(0..10);
            task.done();
            let mut res = task
                .get_result()
                .expect("task is done")
                .collect::<Vec<usize>>();

            // we need to sort since the results are unordered
            res.sort();

            assert!(res.into_iter().zip(1..11).all(|(i1, i2)| i1 == i2));
        }

        #[test]
        fn spmd_task_complains_if_task_is_not_done() {
            let tp = ThreadPool::new(2);
            // let work: Vec<usize> = (0..1000).collect();

            let task = tp.execute(|i: usize| i + 1);

            // will panic if returning an iterator
            let _res = task.get_result().unwrap_err();
        }
    }
}
