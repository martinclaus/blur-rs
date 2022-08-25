use crate::{
    data_type::{Arr2D, Shape2D},
    executor::{Executor, SerialExecutor},
    kernel::Blur,
};
use std::time::Instant;

fn main() {
    let shape = Shape2D(1000, 1000);

    let mut d_in = Arr2D::full(0f64, shape);
    d_in.shape().iter().for_each(|ind| {
        if ((ind[0] as i64) - 499).pow(2) < 10_000 && ((ind[1] as i64) - 499).pow(2) < 10_000 {
            d_in[ind] = 1f64;
        } else {
            d_in[ind] = 0f64;
        }
    });

    let mut d_out = Arr2D::full(0f64, shape);

    let rep = 100;

    let now = Instant::now();
    for _ in 0..rep {
        SerialExecutor::run(Blur, &d_in, &mut d_out);
        SerialExecutor::run(Blur, &d_out, &mut d_in);
    }
    println!("Time elapsed: {}", now.elapsed().as_micros() / 2 / rep);
    (0..1000).for_each(|i| print! {"{:#.2e},", d_in[[500, i]]});
}

mod data_type {
    use std::iter::IntoIterator;
    use std::ops::{Index, IndexMut, Range};
    use std::slice::{Iter, IterMut};

    pub type Ix2 = [usize; 2];
    pub type Item = f64;
    type Range1D = Range<usize>;

    #[derive(Clone, Debug)]
    pub struct Range2D(Range1D, Range1D);

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
        ///
        /// ```
        /// assert_eq!(Blur::map_index([0, 0], [4, 5]), [3, 4]);
        /// assert_eq!(Blur::map_index([2, 1], [4, 5]), [5, 5]);
        /// ```
        fn map_index(k_idx: Ix2, d_idx: Ix2) -> Ix2;
    }

    pub struct Blur;

    const KERNEL_GAUSS: [[Item; 3]; 3] = [
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    ];

    const KERNEL_GAUSS_SHAPE: Shape2D = Shape2D(3, 3);

    impl Blur {
        #[inline]
        fn not_process(idx: Ix2, shape: Shape2D) -> bool {
            idx[0] < Self::shape().0
                || idx[0] >= shape.0 - Self::shape().0
                || idx[1] < Self::shape().1
                || idx[1] >= shape.1 - Self::shape().1
        }
    }

    impl Kernel for Blur {
        fn eval(data: &Arr2D, idx: Ix2) -> Item {
            if Self::not_process(idx, data.shape()) {
                return data[idx];
            }

            let mut sum: Item = 0.0;

            KERNEL_GAUSS.iter().enumerate().for_each(|(j, row)| {
                row.iter()
                    .enumerate()
                    .for_each(|(i, elem)| sum += elem * data[Self::map_index([j, i], idx)]);
            });

            sum
        }

        #[inline]
        fn shape() -> Shape2D {
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
    use crate::data_type::Arr2D;
    use crate::kernel::Kernel;

    pub trait Executor {
        /// Apply kernel operation to all possible indices of `res` and populate it with the results.
        fn run<K: Kernel>(_kernel: K, data: &Arr2D, res: &mut Arr2D);
    }

    /// Simple serial (single-threaded) executor.
    pub struct SerialExecutor;

    impl Executor for SerialExecutor {
        fn run<K: Kernel>(_kernel: K, data: &Arr2D, res: &mut Arr2D) {
            let shape = res.shape();
            shape
                .iter()
                .zip(res.iter_mut())
                .for_each(|(idx, d)| *d = K::eval(data, idx));
        }
    }
}
