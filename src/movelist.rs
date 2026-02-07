use crate::r#move::Move;

/// Move list container for storing generated moves
///
/// Uses a fixed-size array to avoid dynamic allocation during search
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveList {
    moves: [Move; 256],
    len: u16,
}

impl MoveList {
    /// Create an empty move list
    #[inline]
    pub const fn new() -> Self {
        MoveList {
            moves: [Move::null(); 256],
            len: 0,
        }
    }

    /// Add a move to the list
    #[inline]
    pub fn push(&mut self, mv: Move) {
        self.moves[self.len as usize] = mv;
        self.len += 1;
    }

    /// Get the number of moves in the list
    #[inline]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// Check if the list is empty
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear the move list
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Get a move at the given index
    #[inline]
    pub const fn get(&self, index: usize) -> Move {
        self.moves[index]
    }

    /// Get a mutable reference to a move at the given index
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> &mut Move {
        &mut self.moves[index]
    }

    /// Get an iterator over the moves
    #[inline]
    pub fn iter(&self) -> MoveListIter<'_> {
        MoveListIter {
            list: self,
            index: 0,
        }
    }

    /// Remove the move at the given index and shift the rest
    #[inline]
    pub fn remove(&mut self, index: usize) {
        if index < self.len as usize {
            for i in index..self.len as usize - 1 {
                self.moves[i] = self.moves[i + 1];
            }
            self.len -= 1;
        }
    }

    /// Pop the last move from the list
    #[inline]
    pub fn pop(&mut self) -> Option<Move> {
        if self.len > 0 {
            self.len -= 1;
            Some(self.moves[self.len as usize])
        } else {
            None
        }
    }

    /// Swap two moves in the list
    #[inline]
    pub fn swap(&mut self, i: usize, j: usize) {
        if i < self.len as usize && j < self.len as usize {
            self.moves.swap(i, j);
        }
    }

    /// Check if a move exists in the list
    #[inline]
    pub fn contains(&self, mv: Move) -> bool {
        for i in 0..self.len as usize {
            if self.moves[i] == mv {
                return true;
            }
        }
        false
    }

    /// Extend this list with moves from another list
    #[inline]
    pub fn extend(&mut self, other: &MoveList) {
        for i in 0..other.len() {
            self.push(other.get(i));
        }
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = Move;
    type IntoIter = MoveListIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator for MoveList
#[derive(Debug, Clone)]
pub struct MoveListIter<'a> {
    list: &'a MoveList,
    index: usize,
}

impl<'a> Iterator for MoveListIter<'a> {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.list.len() {
            let mv = self.list.get(self.index);
            self.index += 1;
            Some(mv)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.list.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for MoveListIter<'a> {
    fn len(&self) -> usize {
        self.list.len() - self.index
    }
}

/// A move with its score for move ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoredMove {
    pub mv: Move,
    pub score: i32,
}

impl ScoredMove {
    /// Create a new scored move
    #[inline]
    pub fn new(mv: Move, score: i32) -> Self {
        ScoredMove { mv, score }
    }

    /// Get the move
    #[inline]
    pub const fn move_(&self) -> Move {
        self.mv
    }

    /// Get the score
    #[inline]
    pub const fn score(&self) -> i32 {
        self.score
    }
}

/// Move list with scores for move ordering
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoredMoveList {
    moves: Vec<ScoredMove>,
}

impl ScoredMoveList {
    /// Create an empty scored move list
    #[inline]
    pub fn new() -> Self {
        ScoredMoveList { moves: Vec::new() }
    }

    /// Create a scored move list with a given capacity
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        ScoredMoveList {
            moves: Vec::with_capacity(capacity),
        }
    }

    /// Add a scored move to the list
    #[inline]
    pub fn push(&mut self, mv: Move, score: i32) {
        self.moves.push(ScoredMove::new(mv, score));
    }

    /// Get the number of moves in the list
    #[inline]
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// Check if the list is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    /// Clear the move list
    #[inline]
    pub fn clear(&mut self) {
        self.moves.clear();
    }

    /// Get a move at the given index
    #[inline]
    pub fn get(&self, index: usize) -> Move {
        self.moves[index].mv
    }

    /// Sort the moves by score (descending - highest score first)
    #[inline]
    pub fn sort(&mut self) {
        self.moves.sort_by(|a, b| b.score.cmp(&a.score));
    }

    /// Get the move with the highest score and remove it
    #[inline]
    pub fn pop_best(&mut self) -> Option<Move> {
        if self.moves.is_empty() {
            None
        } else {
            let best_idx = self
                .moves
                .iter()
                .enumerate()
                .max_by_key(|(_, sm)| sm.score)
                .map(|(i, _)| i)
                .unwrap();
            Some(self.moves.swap_remove(best_idx).mv)
        }
    }

    /// Convert to a regular move list
    #[inline]
    pub fn to_move_list(&self) -> MoveList {
        let mut ml = MoveList::new();
        for sm in &self.moves {
            ml.push(sm.mv);
        }
        ml
    }
}

impl Default for ScoredMoveList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_list_push() {
        let mut ml = MoveList::new();
        assert_eq!(ml.len(), 0);

        ml.push(Move::new(0, 1));
        assert_eq!(ml.len(), 1);
        assert_eq!(ml.get(0), Move::new(0, 1));
    }

    #[test]
    fn test_move_list_iter() {
        let mut ml = MoveList::new();
        ml.push(Move::new(0, 1));
        ml.push(Move::new(2, 3));
        ml.push(Move::new(4, 5));

        let moves: Vec<Move> = ml.iter().collect();
        assert_eq!(moves.len(), 3);
        assert_eq!(moves[0], Move::new(0, 1));
        assert_eq!(moves[1], Move::new(2, 3));
        assert_eq!(moves[2], Move::new(4, 5));
    }

    #[test]
    fn test_move_list_remove() {
        let mut ml = MoveList::new();
        ml.push(Move::new(0, 1));
        ml.push(Move::new(2, 3));
        ml.push(Move::new(4, 5));

        ml.remove(1);
        assert_eq!(ml.len(), 2);
        assert_eq!(ml.get(0), Move::new(0, 1));
        assert_eq!(ml.get(1), Move::new(4, 5));
    }

    #[test]
    fn test_move_list_pop() {
        let mut ml = MoveList::new();
        ml.push(Move::new(0, 1));
        ml.push(Move::new(2, 3));

        assert_eq!(ml.pop(), Some(Move::new(2, 3)));
        assert_eq!(ml.pop(), Some(Move::new(0, 1)));
        assert_eq!(ml.pop(), None);
    }

    #[test]
    fn test_scored_move_list() {
        let mut sml = ScoredMoveList::new();
        sml.push(Move::new(0, 1), 100);
        sml.push(Move::new(2, 3), 300);
        sml.push(Move::new(4, 5), 200);

        sml.sort();
        assert_eq!(sml.get(0), Move::new(2, 3)); // Score 300
        assert_eq!(sml.get(1), Move::new(4, 5)); // Score 200
        assert_eq!(sml.get(2), Move::new(0, 1)); // Score 100
    }
}
