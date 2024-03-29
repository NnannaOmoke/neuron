use std::{sync::RwLock, path::Path};

use super::*;
use crate::dtype::{self, DType, DTypeType};
use ndarray::IndexLonger;
use std::{borrow::Cow, io::Read};

//this will be the dataset visible to the external users
//we'll implement quite the number of methods for this, hopefully
#[repr(C)]
#[derive(Clone)]
pub(crate) struct BaseDataset<'a> {
    data: BaseMatrix,
    column_names: Cow<'a, [Cow<'a, str>]>,
    std: Option<Vec<DType>>,
    mean: Option<Vec<DType>>,
    mode: Option<Vec<DType>>,
    median: Option<Vec<DType>>,
    percentiles: Option<BaseMatrix>, // Is this inappropriate?
}

impl<'a> BaseDataset<'a> {
    pub fn from_matrix(
        data: BaseMatrix,
        compute_on_creation: bool,
        colnames: Cow<'a, [Cow<'a, str>]>,
    ) -> BaseDataset<'a> {
        if compute_on_creation {
            todo!()
        }
        Self {
            data,
            column_names: colnames,
            std: None,
            mean: None,
            mode: None,
            median: None,
            percentiles: None,
        }
    }
    //we'll have to define a lot more convenience methods for instantiating this, however

    pub fn try_from_csv_reader<R: Read>(
        reader: csv::Reader<R>,
        prefer_precision: bool,
        compute_on_creation: bool,
        colnames: Cow<'a, [Cow<'a, str>]>,
    ) -> Result<BaseDataset<'a>, super::Error> {
        Ok(BaseDataset::from_matrix(
            BaseMatrix::try_from_csv(reader, prefer_precision)?,
            compute_on_creation,
            colnames,
        ))
    }

    pub fn get_col(&self, cindex: usize) -> Option<&[DType]> {
        self.data.get_col(cindex)
    }

    pub fn get_col_mut(&mut self, cindex: usize) -> Option<&mut [DType]> {
        self.data.get_mut_col(cindex)
        // Recalculate cached values
    }

    pub fn get_row(&self, cindex: usize) -> Option<&[DType]> {
        self.data.get_row(cindex)
    }

    pub fn get_row_mut(&mut self, cindex: usize) -> Option<&mut [DType]> {
        self.data.get_mut_row(cindex)
        // Recalculate cached values
    }

    //returns the colum names of the basedataset
    pub fn columns(&'a self) -> Cow<'a, [Cow<'a, str>]> {
        self.column_names.clone()
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
        match row_one {
            // how do we
            Some(row) => {
                //basically print the variants of the enum out
                print!("[");
                for elem in row {
                    print!("{}, ", elem.data_type().display_str())
                }
                print!("]");
            }
            None => println!("[]"),
        }
    }
    //return an estimate of the memory usage of each colunm in bytes
    pub fn memory_usage(&self) -> Vec<usize> {
        let mut return_val = Vec::new();
        let (num_cols, num_rows) = self.data.shape();

        //couldnt find the colunm iterator
        for col_index in 0..num_cols {
            let col = self.data.get_col(col_index).unwrap();
            if num_rows == 0 {
                return_val.push(0);
            } else {
                let size = match &col[0] {
                    DType::Object(_) => {
                        //for strings we want to check the all elements
                        let mut col_size = 0usize;
                        for elem in col {
                            col_size += elem.type_size();
                        }
                        col_size
                    }
                    other => other.type_size() * num_rows,
                };
                return_val.push(size);
            }
        }
        return_val
    }
    //return an estimate of the memory usage of the entire dataset
    pub fn total_memory_usage(&self) -> usize {
        let mut return_val = 0usize;
        let (num_cols, num_rows) = self.data.shape();

        if num_rows != 0 {
            for col_index in 0..num_cols {
                let col = self.data.get_col(col_index).unwrap();
                {
                    let size = match &col[0] {
                        DType::Object(_) => {
                            //for strings we want to check the all elements
                            let mut col_size = 0usize;
                            for elem in col {
                                col_size += elem.type_size();
                            }
                            col_size
                        }
                        other => other.type_size() * num_rows,
                    };
                    return_val += size;
                }
            }
        }
        return_val
    }
    //this will, based on the selection given, return parts of the dataset that have cols that are...
    //of the dtype
    //it will return a read only reference to the current matrix, with maybe a few cols missing?
    pub fn select_dtypes(&self, _include: &[DType], _exlude: Option<&[DType]>) -> &'a Self {
        todo!()
    }
    //returns the number of dimensions of the dataset
    pub fn ndim(&self) -> usize {
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
    pub fn astype(
        &mut self,
        colname: Option<String>,
        dtype: DTypeType,
    ) -> Result<(), dtype::Error> {
        match colname {
            Some(name) => {
                for elem in &mut self[name] {
                    *elem = elem.cast(dtype)?;
                }
            }
            None => {}
        }

        Ok(())
    }
    //returns the first n rows in the dataframe (usually this should be printed out as a table)
    pub fn head(&self, n: Option<usize>) {
        let headers = self
            .column_names
            .iter()
            .map(|name_cow| name_cow.clone().into_owned())
            .collect::<Vec<_>>();
        let data: Vec<Vec<DType>> = self
            .data
            .rows()
            .take(n.unwrap_or(5))
            .map(|x| Vec::from(x.as_slice().unwrap()))
            .collect();
        //we have the headers and the data, now we just use a pretty print macro
        let mut prettytable = prettytable::Table::new();
        prettytable.add_row(headers.into());
        for row in data {
            prettytable.add_row(row.into());
        }
        prettytable.printstd();
    }

    pub fn tail(&self, n: Option<usize>) -> () {
        let _headers = &self.column_names;
        //we need to implement the double ended iterator trait for BaseMatrix for a more efficient implementation of this
        //all this will be replaced when that is done
        let _size = self.data.data.nrows();
    }
    //get the data at a single point
    pub fn display_point(&self, rindex: usize, colname: Option<String>) {
        println!(
            "{}",
            self.data
                .get(rindex, self._get_string_index(&colname.unwrap()))
        )
    }

    //modify the data at a single point
    pub fn modify_point_(&mut self, rindex: usize, colname: Option<String>, new_point: DType) {
        let index = self._get_string_index(&colname.unwrap_or_default());
        let prev = self.data.get_mut(rindex, index);
        *prev = new_point;
    }
    //add a column to the data
    pub fn push_col(&mut self, colname: Cow<'a, str>, slice: &[DType]) {
        self.column_names.to_mut().push(colname);
        self.data.push_col(slice)
    }
    //iterator over column name, data pairs
    pub fn items<'s>(&'s mut self) -> Zip<Iter<Cow<'s, str>>, base_array::ColumnIter<'s>> {
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
    //should pop the last item in the queue
    pub fn pop(&mut self) {
        todo!()
    }
    //applies a function to a given range
    pub fn apply<F>(&mut self, range: Range<usize>, mut function: F)
    where
        F: FnMut(&mut DType) -> (),
    {
        self[range].iter_mut().for_each(|x| function(x))
    }
    //applies a function to a column
    pub fn map<F>(&mut self, colname: String, mut function: F)
    where
        F: FnMut(&mut DType) -> (),
    {
        let col_index = self._get_string_index(&colname);
        self[col_index].iter_mut().for_each(|x| function(x))
    }
    //applies a series of functions to a column
    pub fn pipe<F>(&mut self, colname: String, functions: &mut [F])
    where
        F: FnMut(&mut DType) -> (),
    {
        let col_index = self._get_string_index(&colname);
        self[col_index]
            .iter_mut()
            .for_each(|x| functions.iter_mut().for_each(|f| f(x)))
    }
    //gets the absolute values of all the columns in the dataframe
    pub fn abs(&mut self) {
        self.apply(
            Range {
                start: 0,
                end: self.data.len(),
            },
            DType::abs,
        )
    }

    fn _get_string_index(&self, colname: &String) -> usize {
        self.column_names
            .iter()
            .position(|x| x == colname)
            .expect("Column name was not found")
    }
}

impl<'a> Index<String> for BaseDataset<'a> {
    type Output = [DType];
    fn index(&self, index: String) -> &Self::Output {
        let index = self._get_string_index(&index);
        self.get_col(index).unwrap()
    }
}

impl<'a> Index<usize> for BaseDataset<'a> {
    type Output = [DType];
    fn index(&self, index: usize) -> &Self::Output {
        //this should return a row, no?
        match self.data.get_row(index) {
            Some(slice) => slice,
            None => panic!("Invalid index"),
        }
    }
}

impl<'a> Index<Range<usize>> for BaseDataset<'a> {
    type Output = [DType];
    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.data.data.as_slice().unwrap()[index]
    }
}

impl<'a> IndexMut<Range<usize>> for BaseDataset<'a> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.data.data.as_slice_mut().unwrap()[index]
    }
}

impl<'a> IndexMut<String> for BaseDataset<'a> {
    fn index_mut(&mut self, index: String) -> &mut Self::Output {
        let index = self
            .column_names
            .iter()
            .position(|x| *x == index)
            .expect("Element could not be found");
        self.data
            .get_mut_col(index)
            .expect("This shouldn't be broken")
    }
}

impl<'a> IndexMut<usize> for BaseDataset<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        //this should return a row, no?
        match self.data.get_mut_row(index) {
            Some(slice) => slice,
            None => panic!("Invalid index"),
        }
    }
}
