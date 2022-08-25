use crate::{
    data_type::{Arr2D, Size2D},
    executor::{Executor, SerialExecutor},
    kernel::Blur,
};
use std::time::Instant;

fn main() {
    let size = Size2D(1000, 1000);

    let mut d_in = Arr2D::full(0f64, size);
    d_in.size().iter().for_each(|ind| {
        if ((ind[0] as i64) - 499).pow(2) < 10_000 && ((ind[1] as i64) - 499).pow(2) < 10_000 {
            d_in[ind] = 1f64;
        } else {
            d_in[ind] = 0f64;
        }
    });

    let mut d_out = Arr2D::full(0f64, size);

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

    impl From<Size2D> for Range2D {
        #[inline]
        fn from(size: Size2D) -> Self {
            Range2D(0..size.0, 0..size.1)
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
    pub struct Size2D(pub usize, pub usize);

    impl Size2D {
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

    impl IntoIterator for Size2D {
        type Item = Ix2;
        type IntoIter = Range2D;

        fn into_iter(self) -> Self::IntoIter {
            Range2D::from(self)
        }
    }

    pub struct Arr2D {
        size: Size2D,
        // box slice because of stack size limitations
        data: Box<[Item]>,
    }

    impl Arr2D {
        pub fn full(item: f64, size: Size2D) -> Self {
            Arr2D {
                size,
                data: vec![item; size.0 * size.1].into_boxed_slice(),
            }
        }

        #[inline]
        pub fn size(&self) -> Size2D {
            self.size
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
            &self[self.size.index_into_usize(index)]
        }
    }

    impl IndexMut<Ix2> for Arr2D {
        // row major indexing
        #[inline]
        fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
            let index = self.size().index_into_usize(index);
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

        use super::{Arr2D, Ix2, Size2D};

        #[test]
        fn linear_to_tuple_index() {
            let size = Size2D(2, 3);
            let res: Vec<Ix2> = size
                .iter()
                .enumerate()
                .map(|(i, _ind)| size.usize_into_index(i))
                .collect();
            assert_eq!(res, vec![[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],])
        }

        #[test]
        fn tuple_index_to_linear() {
            let size = Size2D(2, 3);
            let res: Vec<usize> = size.iter().map(|ind| size.index_into_usize(ind)).collect();
            let oracle: Vec<usize> = (0..(size.0 * size.1)).collect();
            assert_eq!(res, oracle)
        }

        #[test]
        fn range2d_iterates_over_all_indices() {
            let s = Size2D(2, 3);
            let res: Vec<Ix2> = s.iter().collect();
            assert_eq!(res, vec![[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]);
        }

        #[test]
        fn arr2d_iter_mut_with_index_in_bounds() {
            let size = Size2D(500, 100);
            let mut data = Arr2D::full(1f64, size);
            let data2 = Arr2D::full(2f64, size);
            size.iter()
                .zip(data.iter_mut())
                .for_each(|(ind, out)| *out = data2[ind]);
            assert!(data.iter().all(|d| *d == 2.0));
        }
    }
}

mod kernel {
    use crate::data_type::{Arr2D, Item, Ix2, Size2D};

    /// Trait for kernel operations.
    pub trait Kernel {
        /// Evaluate kernel operation for given index idx.
        fn eval(data: &Arr2D, idx: Ix2) -> Item;

        /// Return size of the kernel
        fn size() -> Size2D;
    }

    pub struct Blur;

    const KERNEL_GAUSS: [[Item; 3]; 3] = [
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    ];

    const KERNEL_GAUSS_SIZE: Size2D = Size2D(3, 3);

    impl Blur {
        #[inline]
        const fn shift_i() -> usize {
            1
        }

        #[inline]
        const fn shift_j() -> usize {
            1
        }

        #[inline]
        fn not_process(idx: Ix2, shape: Size2D) -> bool {
            idx[0] < Self::size().0
                || idx[0] >= shape.0 - Self::size().0
                || idx[1] < Self::size().1
                || idx[1] >= shape.1 - Self::size().1
        }
    }

    impl Kernel for Blur {
        fn eval(data: &Arr2D, idx: Ix2) -> Item {
            if Self::not_process(idx, data.size()) {
                return data[idx];
            }

            let mut sum: Item = 0.0;

            KERNEL_GAUSS.iter().enumerate().for_each(|(j, row)| {
                row.iter().enumerate().for_each(|(i, elem)| {
                    sum += elem * data[[idx[0] + j - Self::shift_j(), idx[1] + i - Self::shift_i()]]
                });
            });

            sum
        }

        #[inline]
        fn size() -> Size2D {
            KERNEL_GAUSS_SIZE
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
            let size = res.size();
            size.iter()
                .zip(res.iter_mut())
                .for_each(|(idx, d)| *d = K::eval(data, idx));
        }
    }
}
