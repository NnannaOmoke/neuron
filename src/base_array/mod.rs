use crate::{
    dtype::{self, DType, DTypeType},
    *,
};

#[repr(C)]
#[derive(Clone)]
pub(crate) struct BaseMatrix {
    data: Array2<DType>,
}

impl BaseMatrix {
    //@Sporadic Creator you're going to have to save me on this one :(
    // pub(crate) fn from_csv(sep: &str, fname: PathBuf) -> Self{
    // }
    pub(crate) fn transpose(self) -> Self {
        BaseMatrix {
            data: self.data.reversed_axes(),
        }
    }
    pub(crate) fn shape(&self) -> (usize, usize) {
        assert!(self.data.shape().len() == 2);
        (self.data.nrows(), self.data.ncols())
    }
    pub(crate) fn get_col(&self, cindex: usize) -> Option<&[DType]> {
        self.data.column(cindex).to_slice()
    }
    pub(crate) fn get_mut_col(&mut self, cindex: usize) -> Option<&mut [DType]> {
        self.data.column_mut(cindex).into_slice()
    }
    pub(crate) fn get_row(&self, rindex: usize) -> Option<&[DType]> {
        self.data.row(rindex).to_slice()
    }
    pub(crate) fn get_mut_row(&mut self, rindex: usize) -> Option<&mut [DType]> {
        self.data.row_mut(rindex).into_slice()
    }
    pub(crate) fn get(&self, rindex: usize, cindex: usize) -> &DType {
        self.data.get((rindex, cindex)).unwrap()
    }
    pub(crate) fn get_mut(&mut self, rindex: usize, cindex: usize) -> &mut DType {
        self.data.get_mut((rindex, cindex)).unwrap()
    }
    pub(crate) fn cols(&self) -> ColumnIter<'_> {
        ColumnIter {
            inner: self.data.columns().into_iter(),
        }
    }

    pub(crate) fn rows(&self) -> RowIter<'_> {
        RowIter {
            inner: self.data.rows().into_iter(),
        }
    }
}

//@ViableCompute, I want you to implement std::ops::traits for BaseMatrix [add, sub, mult(dot and element wise), div, index(use the get() function)]. When we're done with that we'll write a more userfriendly API that will
//be visible for our users, similar to pandas dataframe
impl AddAssign<&BaseMatrix> for BaseMatrix {
    fn add_assign(&mut self, rhs: &BaseMatrix) {
        assert_eq!(self.shape(), rhs.shape());
        self.data += &rhs.data
    }
}
impl Add<BaseMatrix> for BaseMatrix {
    type Output = BaseMatrix;
    fn add(self, mut rhs: BaseMatrix) -> Self::Output {
        rhs += &self;
        rhs
    }
}

impl SubAssign<BaseMatrix> for BaseMatrix {
    fn sub_assign(&mut self, rhs: BaseMatrix) {
        assert_eq!(self.shape(), rhs.shape());
        self.data -= &rhs.data
    }
}
impl SubAssign<&BaseMatrix> for BaseMatrix {
    fn sub_assign(&mut self, rhs: &BaseMatrix) {
        assert_eq!(self.shape(), rhs.shape());
        self.data -= &rhs.data
    }
}
//Subtraction is not implemented yet

impl Index<usize> for BaseMatrix {
    type Output = [DType];
    fn index(&self, index: usize) -> &Self::Output {
        self.get_row(index).unwrap()
    }
}

// impl<T> std::ops::Mul<T> for BaseMatrix {
//     type Output = BaseMatrix;
//     fn mul(self, rhs: T) -> Self::Output {
//         self
//     }
// }

impl MulAssign<BaseMatrix> for BaseMatrix {
    fn mul_assign(&mut self, rhs: BaseMatrix) {
        self.data *= &rhs.data
    }
}

impl MulAssign<&BaseMatrix> for BaseMatrix {
    fn mul_assign(&mut self, rhs: &BaseMatrix) {
        self.data *= &rhs.data
    }
}
impl Mul<BaseMatrix> for BaseMatrix {
    type Output = BaseMatrix;
    fn mul(self, _rhs: BaseMatrix) -> Self::Output {
        //Uninmplemented because matrix multiplication is not commutative and i'll have to clone()
        self
    }
}

pub(crate) struct ColumnIter<'a> {
    inner: LanesIter<'a, DType, Ix1>,
}

impl<'a> Iterator for ColumnIter<'a> {
    type Item = ArrayBase<ViewRepr<&'a DType>, Dim<[usize; 1]>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub(crate) struct RowIter<'a> {
    inner: LanesIter<'a, DType, Ix1>,
}

impl<'a> Iterator for RowIter<'a> {
    type Item = ArrayBase<ViewRepr<&'a DType>, Dim<[usize; 1]>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

//this will be the dataset visible to the external users
//we'll implement quite the number of methods for this, hopefully
#[repr(C)]
#[derive(Clone)]
pub(crate) struct BaseDataset<'a> {
    data: BaseMatrix,
    column_names: Option<&'a [String]>,
    std: Option<&'a [DType]>,
    mean: Option<&'a [DType]>,
    mode: Option<&'a [DType]>,
    median: Option<&'a [DType]>,
    percentiles: Option<&'a [&'a [DType]]>, //we should put in a vec?
}

impl<'a> BaseDataset<'a> {
    pub fn from_matrix(
        data: BaseMatrix,
        compute_on_creation: bool,
        colnames: Option<&'a [String]>,
    ) -> Self {
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

    //returns the colum names of the basedataset
    pub fn columns(&self) -> &'a [String] {
        match self.column_names {
            Some(names) => names,
            None => &[],
        }
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
    //why would you do this :(
    pub fn deepcopy(&self) -> Self {
        self.clone()
    }
    //returns the first n rows in the dataframe (usually this should be printed out as a table)
    pub fn head(&self, n: Option<usize>) {
        let headers = self.column_names.unwrap_or(&[]);
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
        let _headers = self.column_names.unwrap_or(&[]);
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

    fn _get_string_index(&self, colname: &String) -> usize {
        self.column_names
            .unwrap()
            .iter()
            .position(|x| x == colname)
            .expect("Column name was not found")
    }
}

impl<'a> Index<String> for BaseDataset<'a> {
    type Output = [DType];
    fn index(&self, index: String) -> &Self::Output {
        match self.column_names {
            Some(_) => {
                let index = self._get_string_index(&index);
                self.data.get_row(index).expect("This shouldn't be broken")
            }
            None => panic!("Column names have not been provided!"),
        }
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

impl<'a> IndexMut<String> for BaseDataset<'a> {
    fn index_mut(&mut self, index: String) -> &mut Self::Output {
        match self.column_names {
            Some(names) => {
                let index = names.binary_search(&index).expect("Invalid column name");
                self.data
                    .get_mut_col(index)
                    .expect("This shouldn't be broken")
            }
            None => panic!("Column names have not been provided!"),
        }
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
