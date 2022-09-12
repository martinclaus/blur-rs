//! Data structures to handle 2D arrays of fixed length known at run-time.

use std::iter::IntoIterator;
use std::ops::{Index, IndexMut, Range};
use std::slice::{Iter, IterMut};

use rayon::prelude::*;

pub type Ix2 = [usize; 2];
pub type Item = f64;
type Range1D = Range<usize>;

/// Two-dimensional range of of indices.
///
/// This objects offers an iterator over indices.
#[derive(Clone, Debug)]
pub struct Range2D(pub Range1D, pub Range1D);

impl Range2D {
    pub fn par_iter(&self) -> rayon::vec::IntoIter<Ix2> {
        Range2D(self.0.start..self.0.end, self.1.start..self.1.end)
            .collect::<Vec<_>>()
            .into_par_iter()
    }
}

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

    pub fn par_iter(self) -> rayon::vec::IntoIter<Ix2> {
        Range2D::from(self).par_iter()
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
#[derive(Debug)]
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

    /// Return a slice view of the underlying memory
    // #[inline]
    // pub(crate) fn as_slice(&self) -> &[f64] {
    //     &self.data
    // }

    /// Return a mutable slice view of the underlying memory
    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
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
