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
    print_arr_sample(run_benchmark::<SerialExecutor>());

    // parallel execution
    for nthreads in [8] {
        //(2..=16).step_by(2) {
        println!("RayonExecutor ({} threads)", nthreads);
        print_arr_sample(make_thread_pool(nthreads).install(run_benchmark::<RayonExecutor>));
    }

    println!("ThreadSharedMutableStateExecutor (8 threads)");
    print_arr_sample(run_benchmark::<ThreadSharedMutableStateExecutor>());

    println!("ThreadChannelExecutor (8 threads)");
    print_arr_sample(run_benchmark::<ThreadChannelExecutor>());

    println!("ThreadPoolExecutor (8 threads)");
    print_arr_sample(run_benchmark::<ThreadPoolExecutor>());
}

mod bench {
    use std::{any::type_name, time::Instant};

    use rayon::ThreadPool;

    use crate::{
        data_type::{Arr2D, Shape2D},
        executor::Executor,
        kernel::Blur,
    };

    pub fn make_thread_pool(nthreads: usize) -> ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build()
            .unwrap()
    }

    pub fn print_arr_sample(arr: Arr2D) {
        let i = arr.shape().0 / 2 - arr.shape().1 / 10 - 200;
        print!("Output Sample: ");
        (i..i + 6).for_each(|i| print! {"{:#.2e}, ", arr[[arr.shape().0 / 2, i]]});
        println!();
    }

    pub fn run_benchmark<E: Executor>() -> Arr2D {
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

        let mut d_out = Arr2D::full(0f64, shape);

        let now = Instant::now();
        for _ in 0..rep {
            E::run::<Blur>(&d_in, &mut d_out);
            E::run::<Blur>(&d_out, &mut d_in);
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
    pub struct Range2D(pub Range1D, pub Range1D);

    impl From<Shape2D> for Range2D {
        #[inline]
        fn from(shape: Shape2D) -> Self {
            Range2D(0..shape.0, 0..shape.1)
        }
    }

    impl Iterator for Range2D {
        type Item = Ix2;

        /// Produces indices in row-major ordering
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

    #[derive(Copy, Clone, Debug)]
    pub struct Shape2D(pub usize, pub usize);

    impl Shape2D {
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

    pub struct Arr2D {
        shape: Shape2D,
        // box slice because of stack size limitations
        data: Box<[Item]>,
    }

    impl Arr2D {
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

        #[inline]
        pub fn iter(&self) -> Iter<Item> {
            self.data.iter()
        }

        #[inline]
        pub fn iter_mut(&mut self) -> IterMut<Item> {
            self.data.iter_mut()
        }

        #[inline]
        // FIXME: make this a feature
        pub fn par_iter(&self) -> rayon::slice::Iter<Item> {
            self.data.par_iter()
        }

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
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};

    use crate::data_type::{Arr2D, Item, Range2D};
    use crate::kernel::Kernel;
    use rayon::prelude::*;

    pub trait Executor {
        /// Apply kernel operation to all possible indices of `res` and populate it with the results.
        fn run<K: Kernel>(data: &Arr2D, res: &mut Arr2D);
    }

    /// Simple serial (single-threaded) executor.
    pub struct SerialExecutor;

    impl Executor for SerialExecutor {
        fn run<K: Kernel>(data: &Arr2D, res: &mut Arr2D) {
            let shape = res.shape();
            res.iter_mut()
                // .zip(shape.iter())
                .enumerate() // replace by line above to speed up by factor of ~1.5
                .for_each(|(i, d)| *d = K::eval(data, shape.usize_into_index(i)))
        }
    }

    // FIXME: make this a feature
    /// Executor build upon rayon.
    pub struct RayonExecutor;

    impl Executor for RayonExecutor {
        fn run<K: Kernel>(data: &Arr2D, res: &mut Arr2D) {
            let shape = res.shape();
            // FIXME: use parallel iterator for index
            res.par_iter_mut().enumerate().for_each(|(i, out)| {
                *out = K::eval(data, shape.usize_into_index(i));
            });
        }
    }

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
        fn run<K: Kernel>(data: &Arr2D, res: &mut Arr2D) {
            let nthreads = 8;
            let shape = res.shape();
            let index_range = Self::split_index_range(nthreads, shape.0);

            let res = Arc::new(Mutex::new(res));

            thread::scope(|s| {
                for (i0, i1) in index_range {
                    let res = res.clone();
                    s.spawn(move || {
                        let answer: Vec<Item> = Range2D(i0..i1, 0..shape.1)
                            .map(|idx| K::eval(data, idx))
                            .collect();
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
        fn run<K: Kernel>(data: &Arr2D, res: &mut Arr2D) {
            let nthreads = 8;
            let shape = res.shape();
            let index_range = Self::split_index_range(nthreads, shape.0);

            let (tx, rv) = channel();
            thread::scope(|s| {
                let nthreads = index_range.len();
                for (i0, i1) in index_range {
                    let tx = tx.clone();
                    s.spawn(move || {
                        let answer: Vec<Item> = Range2D(i0..i1, 0..shape.1)
                            .map(|idx| K::eval(data, idx))
                            .collect();
                        tx.send((i0 * shape.1, answer)).unwrap();
                    });
                }
                for _ in 0..nthreads {
                    let (i, answer) = rv.recv().unwrap();
                    res[i..i + answer.len()]
                        .iter_mut()
                        .zip(answer)
                        .for_each(|(r, a)| {
                            *r = a;
                        })
                }
            });
        }
    }

    pub struct ThreadPoolExecutor;

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
    }

    impl Executor for ThreadPoolExecutor {
        fn run<K: Kernel>(data: &Arr2D, res: &mut Arr2D) {
            let nthreads = 8;
            let shape = res.shape();
            let nchunks = nthreads * 5;
            let index_range = Self::split_index_range(nchunks, shape.0);

            // println!("{:#?}", index_range);

            thread::scope(|s| {
                let (tx, rv) = channel();

                // spin up pool of workers
                let mut thread_sender = Vec::<Sender<(usize, usize)>>::with_capacity(nthreads);
                for _ in 0..nthreads {
                    let tx = tx.clone();
                    let (sender, rx) = channel::<(usize, usize)>();
                    thread_sender.push(sender);
                    s.spawn(move || {
                        // will end when channel is disconnected
                        for (i0, i1) in rx {
                            let answer: Vec<Item> = Range2D(i0..i1, 0..shape.1)
                                .map(|idx| K::eval(data, idx))
                                .collect();
                            tx.send(((i0 * shape.1, i1 * shape.1), answer)).unwrap();
                        }
                    });
                }
                drop(tx);

                // feed workers with tasks
                let mut msgs = 0;
                index_range
                    .iter()
                    .zip(thread_sender.into_iter().cycle())
                    .for_each(|(work, thread)| {
                        thread.send(*work).unwrap();
                        msgs += 1;
                    });

                // collect results
                while msgs != 0 {
                    let ((i0, i1), answer) = rv.recv().expect("Oops, thread has panicked");
                    res[i0..i1].iter_mut().zip(answer).for_each(|(r, a)| *r = a);
                    msgs -= 1;
                }
            });
        }
    }
}
