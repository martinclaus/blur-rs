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
    // (0..1000).for_each(|i| print! {"{:#.2e},", d_in[[500, i]]});
}

mod data_type {
    use std::iter::IntoIterator;
    use std::ops::{Index, IndexMut, Range};

    #[derive(Copy, Clone, Debug)]
    pub struct Size2D(pub usize, pub usize);
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

    impl Size2D {
        pub fn iter(self) -> Range2D {
            self.into_iter()
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

        pub fn size(&self) -> Size2D {
            self.size
        }
    }

    impl Index<Ix2> for Arr2D {
        type Output = Item;
        // row major indexing
        #[inline]
        fn index(&self, index: Ix2) -> &Item {
            &self.data[self.size.1 * index[0] + index[1]]
        }
    }

    impl IndexMut<Ix2> for Arr2D {
        // row major indexing
        #[inline]
        fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
            &mut self.data[self.size.1 * index[0] + index[1]]
        }
    }

    #[cfg(test)]
    mod test {
        use super::{Ix2, Size2D};

        #[test]
        fn range2d_iterates_over_all_indices() {
            let s = Size2D(2, 3);
            let res: Vec<Ix2> = s.iter().collect();
            assert_eq!(res, vec![[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]);
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

    impl Blur {
        #[inline]
        const fn shift_i() -> usize {
            1
        }

        #[inline]
        const fn shift_j() -> usize {
            1
        }
    }

    impl Kernel for Blur {
        fn eval(data: &Arr2D, idx: Ix2) -> Item {
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
            Size2D(3, 3)
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
            let kernel_size = K::size();
            let size = res.size();
            let index = size
                .iter()
                // assume symmetric kernel
                .filter(|ind| {
                    ind[0] >= kernel_size.0
                        && ind[0] < size.0 - kernel_size.0
                        && ind[1] >= kernel_size.1
                        && ind[1] < size.1 - kernel_size.1
                });
            index.for_each(|ind| {
                res[ind] = K::eval(data, ind);
            })
        }
    }
}
