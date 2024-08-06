use ndarray::Array1;

use super::*;
use crate::*;

#[derive(Clone, Hash)]
pub enum FillNAStrategy {
    Mean,
    Median,
    Mode,
    Std,
    Value(DType),
}

//this will be the dataset visible to the external users
//we'll implement quite the number of methods for this, hopefully
#[repr(C)]
#[derive(Clone)]
pub struct BaseDataset {
    //TODO: find a suitable constructor for this, deprecate BaseMatrix
    pub(crate) data: BaseMatrix,
    pub(crate) column_names: Vec<String>,
}

impl BaseDataset {
    pub fn from_matrix(data: BaseMatrix, colnames: Vec<String>) -> BaseDataset {
        Self {
            data,
            column_names: colnames,
        }
    }
    //we'll have to define a lot more convenience methods for instantiating this, however
    pub fn try_from_csv_reader<R: Read>(
        reader: csv::Reader<R>,
        prefer_precision: bool,
        colnames: Vec<String>,
    ) -> Result<BaseDataset, super::Error> {
        Ok(BaseDataset::from_matrix(
            BaseMatrix::try_from_csv(reader, prefer_precision)?,
            colnames,
        ))
    }

    pub fn from_csv(
        path: &Path,
        prefer_precision: bool,
        has_headers: bool,
        sep: u8,
    ) -> Result<Self, super::Error> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(has_headers)
            .delimiter(sep)
            .from_path(path)?;
        let mut colnames = vec![];
        if has_headers {
            colnames = reader
                .headers()?
                .iter()
                .map(|slice| String::from(slice).trim().to_string())
                .collect();
        } else {
            //we need to get the length of the reader
            let len = reader.records().next().unwrap().unwrap().iter().count();
            //reset the reader
            reader.seek(Position::new())?;
            (0..len).for_each(|num| {
                colnames.push(format!("Column{}", num));
            });
        }
        Self::try_from_csv_reader(reader, prefer_precision, colnames)
    }
    //iterator over columns
    pub fn cols(&self) -> ColumnIter<'_> {
        self.data.cols()
    }
    //iterator over rows
    pub fn rows(&self) -> RowIter<'_> {
        self.data.rows()
    }
    //mutable iterator over columns
    pub fn cols_mut(&mut self) -> LanesMut<'_, DType, ndarray::Dim<[usize; 1]>> {
        self.data.data.columns_mut()
    }
    //mutable iterator over rows
    pub fn rows_mut(&mut self) -> LanesMut<'_, DType, ndarray::Dim<[usize; 1]>> {
        self.data.data.rows_mut()
    }
    //get a read-only view of a column
    pub fn get_col(&self, cindex: usize) -> ArrayView1<DType> {
        self.data.get_col(cindex)
    }
    //get a mutable view of a column
    pub fn get_col_mut(&mut self, cindex: usize) -> ArrayViewMut1<DType> {
        self.data.get_mut_col(cindex)
        // Recalculate cached values
    }
    //get a read-only view of a row
    pub fn get_row(&self, cindex: usize) -> ArrayView1<DType> {
        self.data.get_row(cindex)
    }
    //get a mutable view of a column
    pub fn get_row_mut(&mut self, cindex: usize) -> ArrayViewMut1<DType> {
        self.data.get_mut_row(cindex)
        // Recalculate cached values
    }
    //returns the colum names of the basedataset
    pub fn columns(&self) -> &Vec<String> {
        &self.column_names
    }
    //this can get a little tricky, but basically we're assuming this
    //every single column has a unique datatype, that everything under it follows
    //and those without a mathematical type will be Objects, which are strings
    //so we have to make sure that on creation of BaseMatrix and BaseDataset from csv files or elsewhere
    //that each column contains elements all of which have a unique datatype
    //if possible, we can cast them lazily...
    //based on this, iterate through the first row and get all the types of the data there
    pub fn dtypes(&self) {
        //iterate through the first column
        //thanks to sporadic creator for wrapping in options. This could have been very tricky otherwise
        let row_one = self.data.get_row(0);
        print!("[");
        for elem in row_one {
            print!("{}, ", elem.data_type().display_str())
        }
        println!("]");
    }
    //return an estimate of the memory usage of the entire dataset
    pub fn total_memory_usage(&self) -> usize {
        let mut value = 0;
        self.rows()
            .for_each(|r| r.iter().for_each(|e| value += e.type_size()));
        value
    }
    //returns the number of dimensions of the dataset
    #[inline]
    pub const fn ndim(&self) -> usize {
        2
    }
    //returns the number of elements in this dataset
    pub fn size(&self) -> usize {
        self.data.data.nrows() * self.data.data.ncols()
    }
    //returns the dimensions of this basedataset
    pub fn shape(&self) -> (usize, usize) {
        self.data.shape()
    }
    //cast all elements in a column to that of another dtype
    pub fn astype(&mut self, colname: &str, dtype: DTypeType) -> Result<(), dtype::Error> {
        let col_index = self._get_string_index(&colname);
        for elem in self.get_col_mut(col_index) {
            *elem = elem.cast(dtype)?;
        }
        Ok(())
    }
    //returns the first n rows in the dataframe (usually this should be printed out as a table)
    pub fn head(&self, n: Option<usize>) {
        let headers = self.column_names.iter().collect::<Vec<_>>();
        let data: Vec<Vec<DType>> = self
            .data
            .rows()
            .take(n.unwrap_or(5))
            .map(|x| x.clone().to_vec())
            .collect();
        //we have the headers and the data, now we just use a pretty print macro
        let mut prettytable = prettytable::Table::new();
        prettytable.add_row(headers.into());
        for row in data {
            prettytable.add_row(row.into());
        }
        prettytable.printstd();
    }
    //returns the last n rows in the dataset
    pub fn tail(&self, n: Option<usize>) {
        let headers = self.column_names.iter().collect::<Vec<_>>();
        let len = self.len();
        let num_rows = if let Some(rows) = n {
            if rows > len {
                5
            } else {
                rows
            }
        } else {
            5
        };
        let mut data = vec![];
        for row in len - 1 - num_rows..len - 1 {
            data.push(Vec::from(self.get_row(row).clone().to_vec()))
        }
        let mut table = prettytable::Table::new();
        table.add_row(headers.into());
        data.iter().for_each(|row| {
            table.add_row(row.into());
        });
        table.printstd();
    }
    //get the data at a single point
    pub fn display_point(&self, rindex: usize, colname: &str) {
        println!(
            "{}",
            self.data
                .get(rindex, self._get_string_index(&colname.to_string()))
        );
    }
    //modify the data at a single point
    pub fn modify_point(&mut self, rindex: usize, colname: &str, new_point: DType) {
        let index = self._get_string_index(colname);
        let prev = self.data.get_mut(rindex, index);
        *prev = new_point;
    }
    //add a column to the data
    pub fn push_col(&mut self, colname: &str, slice: &[DType]) {
        self.column_names.push(colname.to_string());
        self.data.push_col(slice);
    }
    //iterator over column name, data pairs
    pub fn items(&self) -> Zip<Iter<String>, base_array::ColumnIter> {
        zip(self.column_names.iter(), self.data.cols())
    }
    //iterator over row-index, data pairs
    //note that indexes are movable, for now
    pub fn iterrows(&self) -> Zip<Range<usize>, RowIter> {
        zip(0..self.data.len(), self.data.rows())
    }
    //iterator over rows
    pub fn itertuples(&self) -> RowIter {
        self.data.rows()
    }
    //applies a function to a column
    pub fn map<F>(&mut self, colname: &str, function: F)
    where
        F: FnMut(&mut DType),
    {
        let col_index = self._get_string_index(&colname);
        self.get_col_mut(col_index).iter_mut().for_each(function)
    }
    //applies a series of functions to a column
    pub fn pipe<F>(&mut self, colname: &str, functions: &mut [F])
    where
        F: FnMut(&mut DType),
    {
        let col_index = self._get_string_index(&colname);
        self.get_col_mut(col_index)
            .iter_mut()
            .for_each(|x| functions.iter_mut().for_each(|f| f(x)))
    }
    //gets the absolute values of all the columns in the dataframe
    pub fn abs(&mut self) {
        for mut col in self.cols_mut() {
            col.iter_mut().for_each(|x| DType::abs(x))
        }
    }
    //clips the values in a column between certain values
    pub fn clip(&mut self, colname: &str, upper: DType, lower: DType) {
        let col_index = self._get_string_index(&colname);
        for elem in self.get_col_mut(col_index) {
            if *elem > upper {
                *elem = upper.clone();
            }
            if *elem < lower {
                *elem = lower.clone();
            }
        }
    }
    //get the number of non null elements in the column
    pub fn count(&self, colname: &str) -> usize {
        let col_index = self._get_string_index(colname);
        self.get_col(col_index)
            .iter()
            .filter(|x| match x {
                DType::None => false,
                _ => true,
            })
            .count()
    }
    //get the mean of a column
    pub fn mean(&self, colname: &str) -> DType {
        let col_index = self._get_string_index(colname);
        let col = self.get_col(col_index);
        let sum: DType = col
            .iter()
            .filter(|x| match x {
                DType::None | DType::Object(_) => false,
                _ => true,
            })
            .map(|x| x.to_f32().unwrap())
            .sum::<f32>()
            .into();
        let len: DType = (self.len() as f32).into();
        sum.clone() / &len
    }
    //get the median value of a column.
    //fuck this can be hard
    pub fn median(&self, colname: &str) -> DType {
        let col_index = self._get_string_index(colname);
        let mut deepcopy = self.get_col(col_index).clone().to_vec();
        deepcopy.sort();
        deepcopy[self.len() / 2].clone()
    }
    //get the mode of the column, i.e. most occuring element
    pub fn mode(&self, colname: &str) -> DType {
        let col_index = self._get_string_index(colname);
        let col = self.get_col(col_index);
        let counter: Counter<&DType> = Counter::from_iter(col.iter());
        counter.most_common_ordered()[0].0.clone()
    }
    //smallest element in the column
    pub fn min(&self, colname: &str) -> DType {
        let col_index = self._get_string_index(colname);
        self.get_col(col_index)
            .iter()
            .filter(|x| match x {
                DType::None => false,
                _ => true,
            })
            .min()
            .expect("Empty Column")
            .clone()
    }
    //largest element in the column
    pub fn max(&self, colname: &str) -> DType {
        let col_index = self._get_string_index(colname);
        self.get_col(col_index)
            .iter()
            .filter(|x| match x {
                DType::None => false,
                _ => true,
            })
            .max()
            .expect("Empty Column")
            .clone()
    }
    //find the product of all the elements in the column
    pub fn product(&self, colname: &str) -> DType {
        let col_index = self._get_string_index(colname);
        self.get_col(col_index)
            .iter()
            .fold(1u32.into(), |x: DType, y| x * y)
            .clone()
    }
    //find an element at a particular quantile
    pub fn quantile(&self, colname: &str, quantile: f32) -> DType {
        let quantile = (quantile * (self.len() as f32)) as usize;
        let col_index = self._get_string_index(colname);
        let current = self.get_col(col_index);
        let mut deepcopy = current
            .iter()
            .filter(|x| match x {
                DType::None => false,
                _ => true,
            })
            .collect::<Vec<_>>();
        deepcopy.sort();
        deepcopy[quantile].clone()
    }
    pub fn sum(&self, colname: &str) -> DType {
        let col_index = self._get_string_index(colname);
        self.get_col(col_index).sum()
    }
    //the population standard deviation
    pub fn std(&self, colname: &str) -> DType {
        let col_index = self._get_string_index(colname);
        //there will always be cheese
        self.get_col(col_index)
            .map(|x| x.to_f64().unwrap())
            .std(1.0)
            .into()
    }
    //std squared
    pub fn variance(&self, colname: &str) -> DType {
        //std^2
        let std = self.std(colname).clone();
        return std.clone() * std.clone();
    }
    //removes a column
    pub fn drop_col(&mut self, colname: &str) {
        let col_index = self._get_string_index(colname);
        self.column_names.remove(col_index);
        self.data.data.remove_index(Axis(1), col_index);
    }
    //removes a row
    pub fn drop_row(&mut self, row_index: usize) {
        self.data.data.remove_index(Axis(0), row_index);
    }
    pub fn drop_na(&mut self, criteria: Option<usize>, row_first: bool) {
        let delete_marker = if let Some(criteria) = criteria {
            criteria
        } else {
            1
        };
        let mut culprits = Vec::new(); //have to use this cause can't borrow mutably and immutably. should be fine though
                                       //delete if the rows dont match the criteria
        if row_first {
            for (index, row) in self.rows().enumerate() {
                let mut current_count = 0;
                row.iter().for_each(|val| match val {
                    DType::None => current_count += 1,
                    _ => {}
                });
                if current_count >= delete_marker {
                    culprits.push(index);
                }
            }
            for elem in culprits {
                self.drop_row(elem);
            }
        } else {
            for (index, col) in self.cols().enumerate() {
                let mut current_count = 0;
                col.iter().for_each(|val| match val {
                    DType::None => current_count += 1,
                    _ => {}
                });
                if current_count >= delete_marker {
                    culprits.push(index);
                }
            }
            for elem in culprits.iter().rev() {
                self._raw_col_drop(*elem);
            }
        }
    }
    pub fn push_row(&mut self, row: &[DType]) {
        self.data.push_row(row)
    }
    //stack some extra cols
    pub fn vstack(&mut self, other: BaseDataset) {
        //they have the same number of columns
        assert!(self.len() == other.len());
        let mut empty = Array2::from_elem((self.len(), 0), DType::None);
        empty.append(Axis(1), self.data.data.view()).unwrap();
        empty.append(Axis(1), other.data.data.view()).unwrap();
        self.data = BaseMatrix { data: empty };
        self.column_names
            .extend(other.column_names.iter().map(|x| x.clone()));
    }
    //stack some extra rows
    //we want to add the append the other BaseDataset row (i.e. add the external dataset to the bottom of the array) wise. We have to assert that their lengths are the same
    pub fn hstack(&mut self, other: BaseDataset) {
        //they have the same number of rows
        assert!(self.column_names.len() == other.column_names.len());
        let mut empty = Array2::from_elem((0, self.column_names.len()), DType::None);
        empty.append(Axis(0), self.data.data.view()).unwrap();
        empty.append(Axis(0), other.data.data.view()).unwrap();
        self.data = BaseMatrix { data: empty };
    }

    //number of rows in the dataset
    pub fn len(&self) -> usize {
        self.data.len()
    }
    //merge 2 cols together
    pub fn merge_col<F>(&mut self, left: &str, right: &str, function: F, result_name: &str)
    where
        F: Fn(&DType, &DType) -> DType,
    {
        let left_index = self._get_string_index(left);
        let right_index = self._get_string_index(right);
        //merge into the left_index
        let right = self.get_col(right_index).to_owned();
        let mut left = self.get_col_mut(left_index);
        //the cols are of the same length, so we can do this
        for (index, elem) in left.iter_mut().enumerate() {
            *elem = function(&*elem, &right[index])
        }
        self.column_names[left_index] = result_name.to_string();
        self._raw_col_drop(right_index)
    }

    pub fn value_counts(&self, colname: &str) -> HashMap<&DType, usize> {
        let index = self._get_string_index(colname);
        let mappings = Counter::from_iter(self.get_col(index).into_iter());
        mappings.into_map()
    }

    pub fn nunique(&self, colname: &str) -> usize {
        self.value_counts(colname).len()
    }

    pub fn unique(&self, colname: &str) -> Vec<DType> {
        self.value_counts(colname)
            .keys()
            .into_iter()
            .map(|x| x.clone().clone())
            .collect::<Vec<DType>>()
    }

    pub fn fillna(&mut self, colname: &str, strategy: FillNAStrategy) {
        match strategy {
            FillNAStrategy::Mean => {
                let mean = self.mean(colname);
                let index = self._get_string_index(colname);
                self.get_col_mut(index)
                    .iter_mut()
                    .filter(|x| match x {
                        DType::None => true,
                        _ => false,
                    })
                    .for_each(|x| *x = mean.clone())
            }
            FillNAStrategy::Mode => {
                let mode = self.mode(colname);
                let index = self._get_string_index(colname);
                self.get_col_mut(index)
                    .iter_mut()
                    .filter(|x| match x {
                        DType::None => true,
                        _ => false,
                    })
                    .for_each(|x| *x = mode.clone())
            }
            FillNAStrategy::Median => {
                let median = self.median(colname);
                let index = self._get_string_index(colname);
                self.get_col_mut(index)
                    .iter_mut()
                    .filter(|x| match x {
                        DType::None => true,
                        _ => false,
                    })
                    .for_each(|x| *x = median.clone())
            }
            FillNAStrategy::Std => {
                let std = self.std(colname);
                let index = self._get_string_index(colname);
                self.get_col_mut(index)
                    .iter_mut()
                    .filter(|x| match x {
                        DType::None => true,
                        _ => false,
                    })
                    .for_each(|x| *x = std.clone())
            }
            FillNAStrategy::Value(var) => {
                let index = self._get_string_index(colname);
                self.get_col_mut(index)
                    .iter_mut()
                    .filter(|x| match x {
                        DType::None => true,
                        _ => false,
                    })
                    .for_each(|x| *x = var.clone())
            }
        }
    }

    pub fn nrows(&self) -> usize {
        self.data.data.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.data.data.ncols()
    }

    pub fn sort_by(&mut self, colname: &str) {
        let index = self._get_string_index(colname);
        let mut indices = Vec::from_iter(0..self.len());
        indices.sort_by_cached_key(|x| self.get(*x, index));
        let mut data = Array2::from_elem(self.data.shape(), DType::zero());
        for row_index in 0..self.data.len() {
            data.row_mut(row_index)
                .assign(&self.get_row(indices[row_index]));
        }
        self.data = BaseMatrix { data };
    }

    pub(crate) fn get(&self, rowindex: usize, colindex: usize) -> &DType {
        self.data.get(rowindex, colindex)
    }

    pub(crate) fn _raw_col_drop(&mut self, col_index: usize) {
        self.column_names.remove(col_index);
        self.data.data.remove_index(Axis(1), col_index)
    }

    pub(crate) fn _get_string_index(&self, colname: &str) -> usize {
        self.column_names
            .iter()
            .position(|x| x == colname)
            .expect("Column name was not found")
    }
    pub(crate) fn get_ndarray(&self) -> Array2<DType> {
        self.data.data.clone()
    }
    pub(crate) fn into_f64_array_without_target(&self, target: usize) -> Array2<f64> {
        let mut data = Array2::from_elem((self.data.shape().0, self.shape().1 - 1), 0f64);
        for (index, col) in self.cols().enumerate() {
            if index == target {
                continue;
            }
            data.column_mut(index)
                .assign(&col.map(|x| x.to_f64().unwrap()).view());
        }
        data
    }
    pub(crate) fn into_f64_array(&self) -> Array2<f64> {
        let mut data = Array2::from_elem((self.shape().0, self.data.shape().1), 0f64);
        self.rows().enumerate().for_each(|(index, row_view)| {
            data.row_mut(index)
                .assign(&row_view.map(|x| x.to_f64().unwrap()).view());
        });
        data
    }
    pub(crate) fn get_first_nononetype(&self, col: &str) -> DType {
        let index = self._get_string_index(col);
        self.get_col(index)
            .iter()
            .find(|t| match t {
                DType::None => false,
                _ => true,
            })
            .unwrap()
            .clone()
    }
}

impl Default for BaseDataset {
    fn default() -> Self {
        BaseDataset {
            data: BaseMatrix::default(),
            column_names: Vec::default(),
        }
    }
}

impl Index<(usize, usize)> for BaseDataset {
    type Output = DType;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
    }
}

impl IndexMut<(usize, usize)> for BaseDataset {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.data.data.get_mut(index).unwrap()
    }
}
