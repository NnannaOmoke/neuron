use std::{sync::RwLock, path::Path, collections::HashSet};

use super::*;
use crate::dtype::{self, DType, DTypeType};
use ndarray::{IndexLonger, iter::{Axes, Indices, LanesMut}, s};
use std::{borrow::Cow, io::Read, mem::transmute};

//this will be the dataset visible to the external users
//we'll implement quite the number of methods for this, hopefully
#[repr(C)]
#[derive(Clone)]
pub(crate) struct BaseDataset<'a> {
    data: BaseMatrix,
    column_names: Cow<'a, [Cow<'a, str>]>,
}

impl<'a> BaseDataset<'a> {
    pub fn from_matrix(
        data: BaseMatrix,
        compute_on_creation: bool,
        colnames: Cow<'a, [Cow<'a, str>]>,
    ) -> BaseDataset<'a> {
        Self {
            data,
            column_names: colnames,
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
    //iterator over columns 
    pub fn cols(&self) -> ColumnIter<'_>{
        self.data.cols()
    }
    //iterator over rows
    pub fn rows(&self) -> RowIter<'_>{
        self.data.rows()
    }
    //mutable iterator over columns
    pub fn cols_mut(&mut self) -> LanesMut<'_, DType, ndarray::Dim<[usize; 1]>>{
        self.data.data.columns_mut()
    }
    //mutable iterator over rows
    pub fn rows_mut(&mut self) -> LanesMut<'_, DType, ndarray::Dim<[usize; 1]>>{
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
        print!("[");
        for elem in row_one {
            print!("{}, ", elem.data_type().display_str())
        }
        println!("]");
    }
    //return an estimate of the memory usage of each colunm in bytes
    pub fn memory_usage(&self) -> Vec<usize> {
        let mut return_val = Vec::new();
        let (num_cols, num_rows) = self.data.shape();
        //couldnt find the colunm iterator
        for col_index in 0..num_cols {
            let col = self.data.get_col(col_index);
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
                let col = self.data.get_col(col_index);
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
    pub fn astype(
        &mut self,
        colname: String,
        dtype: DTypeType,
    ) -> Result<(), dtype::Error> {
        let col_index = self._get_string_index(&colname);
        for elem in self.get_col_mut(col_index){
            *elem = elem.cast(dtype)?;
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
    //returns the last elements in the dataset
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
    //should pop the last col in the queue
    pub fn pop(&mut self) -> Option<&[DType]>{
        todo!()
    }
    
    //applies a function to a given range
    // pub fn apply<F>(&mut self, range: Range<usize>, mut function: F)
    // where
    //     F: FnMut(&mut DType) -> (),
    // {
    //     self.data[range].iter_mut().for_each(|x| function(x))
    // }

    //applies a function to a column
    pub fn map<F>(&mut self, colname: String, mut function: F)
    where
        F: FnMut(&mut DType) -> (),
    {
        let col_index = self._get_string_index(&colname);
        self.get_col_mut(col_index).iter_mut().for_each(|x| function(x))
    }
    //applies a series of functions to a column
    pub fn pipe<F>(&mut self, colname: String, functions: &mut [F])
    where
        F: FnMut(&mut DType) -> (),
    {
        let col_index = self._get_string_index(&colname);
        self.get_col_mut(col_index)
            .iter_mut()
            .for_each(|x| functions.iter_mut().for_each(|f| f(x)))
    }
    //gets the absolute values of all the columns in the dataframe
    pub fn abs(&mut self) {
        for mut col in self.cols_mut(){
            col.iter_mut().for_each(|x| DType::abs(x))
        }
    }
    //clips the values in a column between certain values
    pub fn clip(&mut self, colname: String, upper: DType, lower: DType){
        let col_index = self._get_string_index(&colname);
        for elem in self.get_col_mut(col_index){
            if *elem > upper{
                *elem  = upper.clone();
            }
            if *elem < lower{
                *elem = lower.clone();
            }
        }
    }
    //get the number of non null elements in the column
    pub fn count(&self, colname: &String) -> usize{
        let col_index = self._get_string_index(colname);
        self.get_col(col_index).iter().filter(|x| match x{DType::None => false, _ => true}).count()
    }
    //get the mean of a column
    pub fn mean(&self, colname: &String) -> DType{
        let col_index = self._get_string_index(colname);
        let sum = self.get_col(col_index).sum();
        let len: DType = (self.len() as f32).into();
        sum/len
    }
    //get the median value of a column.
    //fuck this can be hard
    pub fn median(&self, colname: &String) -> DType{
        let col_index = self._get_string_index(colname);
        let mut deepcopy = self.get_col(col_index).clone().to_vec();
        deepcopy.sort();
        deepcopy[self.len()/2].clone()
    }
    //get the mode of the column, i.e. most occuring element
    pub fn mode(&self, colname: &String) -> DType{
        let col_index = self._get_string_index(colname);
        todo!()
    }
    //smallest element in the column
    pub fn min(&self, colname: &String) -> DType{
        let col_index = self._get_string_index(colname);
        self.get_col(col_index).iter().min().expect("Empty Column").clone()
    }
    //largest element in the column
    pub fn max(&self, colname: &String) -> DType{
        let col_index = self._get_string_index(colname);
        self.get_col(col_index).iter().max().expect("Empty Column").clone()
    }
    //find the product of all the elements in the column
    pub fn product(&self, colname: &String) -> DType{
        let col_index = self._get_string_index(colname);
        self.get_col(col_index).iter().fold(1u32.into(), |x: DType, y| x * y).clone()
    }
    //find an element at a particular quantile
    pub fn quantile(&self, colname: &String, quantile: f32) -> DType{
        let quantile = (quantile * (self.len() as f32)) as usize;
        let col_index = self._get_string_index(colname);
        let mut deepcopy = self.get_col(col_index).clone().to_vec();
        deepcopy.sort();
        deepcopy[quantile].clone()
    }
    pub fn sum(&self, colname: &String) -> DType{
        let col_index = self._get_string_index(colname);
        self.get_col(col_index).sum()
    }
    //for all these methods,we need to make f32 hashable 
    pub fn std(&self, colname: &String) -> DType{
        todo!()
    }
    pub fn variance(&self, colname: &String) -> DType{
        //std^2
        todo!()
    }
    pub fn nunique(&self, colname: &String) -> DType{
        todo!()
    }
    pub fn value_counts(&self, colname: &String) -> &[(String, DType)]{
        todo!()
    }
    //removes a column
    pub fn drop_col(&mut self, colname: &String){
        let col_index = self._get_string_index(colname);
        self.data.data.remove_index(Axis(1), col_index);
    }
    //removes a row
    pub fn drop_row(&mut self, row_index: usize){
        self.data.data.remove_index(Axis(0), row_index);
    }
    pub fn drop_na(&mut self, criteria: Option<usize>, row_first: bool) {
        let delete_marker = match criteria{
            Some(marker) => marker,
            None => 1,
        };
        let mut culprits = Vec::new();//have to use this cause can't borrow mutably and immutably. should be fine though
        //delete if the rows dont match the criteria
        if row_first{
            for (index, row) in self.rows().enumerate(){
                let mut current_count = 0;
                row.into_iter().for_each(|val| {
                    match val{
                        DType::None => current_count += 1,
                        _ => {}
                    }
                });
                if current_count >= delete_marker{
                   culprits.push(index);
                }

            }
            for elem in culprits{
                self.drop_row(elem);
            }
        }
        else{
            for(index, col) in self.cols().enumerate(){
                let mut current_count = 0;
                col.into_iter().for_each(|val| {
                    match val{
                        DType::None => current_count += 1,
                        _ => {}
                    }
                });
                if current_count >= delete_marker{
                    culprits.push(index);
                }
            }
            for elem in culprits{
                self._raw_col_drop(elem);
            }
        }
        
    }
    pub fn transpose(&mut self){
        todo!()
    }
    pub fn push_row(&mut self, row: &[DType]){
        self.data.push_row(row)
    }
    //stack some extra rows 
    pub fn vstack(&mut self, other: BaseDataset){
        //they have the same number of columns
        assert!(self.len() == other.len());
        self.data.data.append(Axis(0), other.data.data.view()).expect("Conjoining the arrays failed")
    }

    //stack some extra columns
    //we want to add the append the other BaseDataset columns wise. We have to assert that their lengths are the same
    pub fn hstack(&mut self, other: BaseDataset){
        //they have the same number of rows
         assert!(self.column_names.len() == other.column_names.len());
        self.data.data.append(Axis(1), other.data.data.view()).expect("Conjoining the arrays vertically failed!")
    }

    //number of rows in the dataset
    pub fn len(&self) -> usize{
        self.data.len()
    }
    fn _raw_col_drop(&mut self, col_index: usize){
        self.data.data.remove_index(Axis(1), col_index)
    } 
    fn _get_string_index(&self, colname: &String) -> usize {
        self.column_names
            .iter()
            .position(|x| x == colname)
            .expect("Column name was not found")
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
