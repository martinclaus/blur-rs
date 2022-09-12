//! Kernel based computations on 2D arrays.

use crate::data::{Arr2D, Item, Ix2, Shape2D};

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
    /// ```rust
    /// assert_eq!(Blur::map_index([0, 0], [4, 5]), [3, 4]);
    /// assert_eq!(Blur::map_index([2, 1], [4, 5]), [5, 5]);
    /// ```
    fn map_index(k_idx: Ix2, d_idx: Ix2) -> Ix2;
}

#[derive(Copy, Clone, Debug)]
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
