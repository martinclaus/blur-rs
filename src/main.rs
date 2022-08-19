use crate::{
    data_type::{Arr2D, Size2D},
    executor::{Executor, SerialExecutor},
    kernel::Blur,
};
use std::time::Instant;

fn main() {
    let size = Size2D(1000, 1000);

    let mut d_in = Arr2D::full(0f64, size);
    d_in.size().into_iter().for_each(|ind| {
        if (ind[0] - 500).pow(2) < 10_000 && (ind[1] - 500).pow(2) < 10_000 {
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
    (0..1000).for_each(|i| print! {"{},", d_in[[500, i]]});
}

mod data_type {
    use std::iter::{IntoIterator, Iterator};
    use std::ops::{Index, IndexMut};

    #[derive(Copy, Clone)]
    pub struct Size2D(pub usize, pub usize);
    pub type Ix2 = [usize; 2];
    pub type Item = f64;

    impl Size2D {
        pub fn into_indexer(self, halo_y: usize, halo_x: usize) -> Indexer2D {
            Indexer2D {
                start_x: halo_x,
                end_x: self.1 - halo_x,
                start_y: halo_y,
                end_y: self.0 - halo_y,
            }
        }
    }

    impl IntoIterator for Size2D {
        type Item = Ix2;
        type IntoIter = IndexIterator2D;

        fn into_iter(self) -> Self::IntoIter {
            self.into_indexer(0, 0).into_iter()
        }
    }

    pub struct Indexer2D {
        start_x: usize,
        end_x: usize,
        start_y: usize,
        end_y: usize,
    }

    impl IntoIterator for Indexer2D {
        type Item = Ix2;
        type IntoIter = IndexIterator2D;

        fn into_iter(self) -> Self::IntoIter {
            let (start_x, start_y) = (self.start_x, self.start_y);
            IndexIterator2D {
                indexer: self,
                i: start_x,
                j: start_y,
            }
        }
    }

    pub struct IndexIterator2D {
        indexer: Indexer2D,
        i: usize,
        j: usize,
    }

    impl Iterator for IndexIterator2D {
        type Item = Ix2;

        fn next(&mut self) -> Option<Self::Item> {
            let ret = {
                if self.j < self.indexer.end_y {
                    Some([self.j, self.i])
                } else {
                    return None;
                }
            };

            self.i += 1;

            if self.i >= self.indexer.end_x {
                self.i = self.indexer.start_x;
                self.j += 1;
            }

            ret
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
        fn shift_i() -> usize {
            1
        }

        fn shift_j() -> usize {
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
        fn run<K: Kernel>(kernel: K, data: &Arr2D, res: &mut Arr2D);
    }

    /// Simple serial (single-threaded) executor.
    pub struct SerialExecutor;

    impl Executor for SerialExecutor {
        fn run<K: Kernel>(_kernel: K, data: &Arr2D, res: &mut Arr2D) {
            let kernel_size = K::size();
            let index = data
                .size()
                // assume symmetric kernel
                .into_indexer(kernel_size.0 / 2, kernel_size.1 / 2)
                .into_iter();
            index.for_each(|ind| {
                res[ind] = K::eval(data, ind);
            })
        }
    }
}
