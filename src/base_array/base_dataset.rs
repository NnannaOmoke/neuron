use ndarray::Array1;

use super::*;
use crate::*;
//this will be the dataset visible to the external users
//we'll implement quite the number of methods for this, hopefully
#[repr(C)]
#[derive(Clone)]
pub(crate) struct BaseDataset {
    data: BaseMatrix,
    column_names: Vec<String>,
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
            (0..len).into_iter().for_each(|num| {
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
    pub fn get_col(&self, cindex: usize) -> ArrayView1<'_, DType> {
        self.data.get_col(cindex)
    }
    //get a mutable view of a column
    pub fn get_col_mut(&mut self, cindex: usize) -> ArrayViewMut1<'_, DType> {
        self.data.get_mut_col(cindex)
        // Recalculate cached values
    }
    //get a read-only view of a row
    pub fn get_row(&self, cindex: usize) -> ArrayView1<'_, DType> {
        self.data.get_row(cindex)
    }
    //get a mutable view of a column
    pub fn get_row_mut(&mut self, cindex: usize) -> ArrayViewMut1<'_, DType> {
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
    pub fn items<'s>(&'s mut self) -> Zip<Iter<String>, base_array::ColumnIter<'s>> {
        zip(self.column_names.iter(), self.data.cols())
    }
    //iterator over row-index, data pairs
    //note that indexes are movable, for now
    pub fn iterrows(&self) -> Zip<Range<usize>, RowIter<'_>> {
        zip(0..self.data.len(), self.data.rows())
    }
    //iterator over rows
    pub fn itertuples(&self) -> RowIter<'_> {
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
        let sum = self.get_col(col_index).sum();
        let len: DType = (self.len() as f32).into();
        sum / len
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
        let dtypetype = DTypeType::from(self.get_col(col_index).first().unwrap());
        match dtypetype {
            DTypeType::None => {
                panic!()
            }
            DTypeType::F32 | DTypeType::F64 | DTypeType::U32 | DTypeType::U64 => {
                let counter = self
                    .get_col(col_index)
                    .iter()
                    .filter(|x| match x {
                        DType::None => false,
                        _ => true,
                    })
                    .map(|val| match val {
                        DType::F32(val) => NotNan::from_f64(*val as f64),
                        DType::F64(val) => NotNan::from_f64(*val as f64),
                        DType::U32(val) => NotNan::from_f64(*val as f64),
                        DType::U64(val) => NotNan::from_f64(*val as f64),
                        _ => unimplemented!(),
                    })
                    .collect::<Counter<_>>();
                let highest_val: Option<NotNan<f64>> = counter.most_common()[0].0;
                highest_val.unwrap().to_f64().unwrap().into()
            }
            DTypeType::Object => {
                let counter = self
                    .get_col(col_index)
                    .iter()
                    .filter(|x| match x {
                        DType::None => false,
                        _ => true,
                    })
                    .map(|val| match val {
                        DType::Object(val) => val.clone(),
                        _ => unimplemented!(),
                    })
                    .collect::<Counter<_>>();
                counter.most_common()[0].0.clone().into()
            }
        }
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
            .map(|x| match x {
                DType::F32(var) => *var as f64,
                DType::F64(var) => *var,
                DType::U32(var) => *var as f64,
                DType::U64(var) => *var as f64,
                _ => panic!("{}", dtype::ERR_MSG_INCOMPAT_TYPES),
            })
            .std(1.0)
            .into()
    }
    //std squared
    pub fn variance(&self, colname: &str) -> DType {
        //std^2
        let std = self.std(colname);
        return (&std * &std).into();
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
                row.into_iter().for_each(|val| match val {
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
                col.into_iter().for_each(|val| match val {
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
    pub fn transpose(&mut self) {
        todo!()
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
        let mut merged = Array1::from_elem((self.len(),), DType::None);
        for (index, (left, right)) in zip(
            self.get_col(left_index).iter(),
            self.get_col(right_index).iter(),
        )
        .enumerate()
        {
            merged[index] = function(left, right);
        }
        self.data
            .data
            .push_column(merged.view())
            .expect("Unexpected Shape on merge");
        self._raw_col_drop(left_index);
        self._raw_col_drop(right_index);
        self.column_names.push(result_name.to_string());
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
    pub(crate) fn into_f64_array(&self, target: usize) -> Array2<f64> {
        let mut data = Array2::from_elem((self.data.shape().0, 0usize), 0f64);
        for (index, col) in self.cols().enumerate() {
            if index == target {
                continue;
            }
            data.push_column(col.map(|x| x.to_f64().unwrap()).view())
                .unwrap();
        }
        println!("{:?}", data.shape());
        data
    }
}

// impl<'a> Index<String> for BaseDataset<'a> {
//     type Output = ArrayView1<'a, DType>;
//     fn index(&self, index: String) -> &Self::Output {
//         let index = self._get_string_index(&index);
//         &self.data.get_col(index)
//     }
// }

// impl<'a> Index<usize> for BaseDataset<'a> {
//     type Output = [DType];
//     fn index(&self, index: usize) -> &Self::Output {
//         //this should return a row, no?
//        &self.data[index]
//     }
// }

// impl<'a> Index<Range<usize>> for BaseDataset<'a> {
//     type Output = [DType];
//     fn index(&self, index: Range<usize>) -> &Self::Output {
//         &self.data.data.as_slice().unwrap()[index]
//     }
// }

// impl<'a> IndexMut<Range<usize>> for BaseDataset<'a> {
//     fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
//         &mut self.data.data.as_slice_mut().unwrap()[index]
//     }
// }

// impl<'a> IndexMut<String> for BaseDataset<'a> {
//     fn index_mut(&mut self, index: String) -> &mut Self::Output {
//         let index = self
//             .column_names
//             .iter()
//             .position(|x| *x == index)
//             .expect("Element could not be found");
//         todo!()

//     }
// }

// impl<'a> IndexMut<usize> for BaseDataset<'a> {
//     fn index_mut(&mut self, index: usize) -> &mut Self::Output {
//         //this should return a row, no?
//         match self.data.get_mut_row(index) {
//             Some(slice) => slice,
//             None => panic!("Invalid index"),
//         }
//     }
// }
