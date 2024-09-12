use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Ident, Lit};

#[proc_macro]
pub fn dtype(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Lit);
    let dtype = match input {
        Lit::Str(val) => {
            quote! {
                ::neuron::dtype::DType::Object(Box::new(#val.to_string()))
            }
        }
        Lit::ByteStr(val) => {
            quote! {
                ::neuron::dtype::DType::Object(Box::new(::std::string::String::from_utf8(#val)))
            }
        }
        Lit::Char(val) => {
            quote! {
                ::neuron::dtype::DType::Object(Box::new(::std::string::String::from(#val)))
            }
        }
        Lit::Int(val) => {
            //find if it's negative, if it is, make it float, else unsigned int
            let value = val.base10_parse::<f64>().unwrap();
            if value < 0f64 {
                quote! {
                    ::neuron::dtype::DType::F64(#value as f64)
                }
            } else {
                quote! {
                    ::neuron::dtype::DType::U64(#value as u64)
                }
            }
        }
        Lit::Float(val) => {
            //remember, the float CANNOT be NaN or weird in any manner shape or form
            let val = val.base10_parse::<f64>().unwrap();
            if val.is_normal() {
                quote! {
                    ::neuron::dtype::DType::F64(#val)
                }
            } else {
                quote! {
                    ::neuron::dtype::DType::None
                }
            }
        }
        Lit::Bool(val) => {
            quote! {
                ::neuron::dtype::DType::U32(#val as u32)
            }
        }
        _ => quote! {::neuron::dtype::DType::None},
    };
    dtype.into()
}

#[proc_macro_derive(CrossValidator, attributes(validate))]
pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let validator_name = format!("{}Validator", name);
    let validator_ident = Ident::new(&validator_name, name.span());
    let fields = if let syn::Data::Struct(syn::DataStruct {
        fields: syn::Fields::Named(syn::FieldsNamed { ref named, .. }),
        ..
    }) = input.data
    {
        named
    } else {
        unimplemented!()
    };

    let mut vec_takers = vec![];
    let mut better = fields
        .iter()
        .map(|f| {
            let n = &f.ident;
            let ty = &f.ty;
            for attr in &f.attrs {
                if attr.path().get_ident().unwrap() == &Ident::new("validate", name.span()) {
                    vec_takers.push(f.clone());
                    return quote! {
                        #n: std::vec::Vec<#ty>
                    };
                }
            }
            quote! {
                #n: #ty
            }
        })
        .collect::<Vec<_>>();
    better.push(quote! {
       best_estimator: Option<#name>,
        results: Option<::std::vec::Vec<(::std::string::String, f64)>>
    });
    let len = vec_takers.len();
    if len == 0 {
        panic!("No field can be cross-validated");
    }
    let methods = vec_takers.iter().map(|f| {
        let n = &f.ident;
        let ty = &f.ty;
        quote! {
            pub fn #n(self, #n: std::vec::Vec<#ty>) -> Self{
                Self{
                    #n, ..self
                }
            }
        }
    });
    let mut into_validator_iterator = fields
        .iter()
        .map(|f| {
            let n = &f.ident;
            for attr in &f.attrs {
                if attr.path().get_ident().unwrap() == &Ident::new("validate", name.span()) {
                    return quote! {
                        #n:  std::vec![self.#n]
                    };
                }
            }
            quote! {
                #n: self.#n
            }
        })
        .collect::<Vec<_>>();
    into_validator_iterator.push(quote! {
       best_estimator: None,
        results: None,
    });
    let into_validator = quote! {
        //firstly we want to make a fn called into validator with signature
        pub fn into_validator(self) -> #validator_ident{
            //define an iterator that goes over all the fields and maps them to the validator
            //those with the validator attribute get to have their values wrapped in a vec![], like so
            #validator_ident{
                #(#into_validator_iterator),*
            }
        }
    };
    let struct_cross_val_gen = fields
        .iter()
        .map(|f| {
            let n = &f.ident;
            for attr in &f.attrs {
                if attr.path().get_ident().unwrap() == &Ident::new("validate", name.span()) {
                    return quote! {
                        #n: #n.clone()
                    };
                }
            }
            quote! {
                #n: self.#n.clone()
            }
        })
        .collect::<Vec<_>>();

    let multiple_fors =
        vec_takers
            .iter()
            .enumerate()
            .rev()
            .fold(quote!(), |accum, (index, field)| {
                let ident = field.ident.clone().unwrap();
                if index == len - 1 {
                    quote! {
                        for #ident in &self.#ident{
                            let current = #name{
                                #(#struct_cross_val_gen),*
                            };
                            container.push(current);

                        }
                    }
                } else {
                    quote! {
                        for #ident in &self.#ident{
                            #accum
                        }
                    }
                }
            });
    let res = quote! {
        use rayon::iter::ParallelIterator;
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefMutIterator;
        impl #name{
            #into_validator
        }

        pub struct #validator_ident{
            #(#better),*
        }

        impl #validator_ident{
            #(#methods)*

            //now we write the fit method
            //this time, fit WILL consume the basedataset
            pub fn fit(&mut self, dataset: ::neuron::base_array::BaseDataset, target: &str, evaluation: fn(::ndarray::ArrayView1<f64>, ::ndarray::ArrayView1<f64>) -> std::vec::Vec<f64>, reverse: bool){
                //so basically we want to make multiple copies of the estimators, which means another traipsing of the fields
                let mut container = ::std::vec![];
                #multiple_fors
                let indices = ::std::sync::Arc::new(::std::sync::RwLock::new(::std::vec::Vec::from_iter((0 .. container.len()).map(|_| 0f64))));
                container.par_iter_mut().enumerate().for_each(|(index, estimator)| {
                    estimator.fit(&dataset, target);
                    let result = estimator.evaluate(evaluation);
                    if result.len() == 1{
                        let lock = indices.clone();
                        let mut lock = lock.write().unwrap();
                        lock[index] = result[0];
                    } else {
                        unimplemented!()
                    }
                });
                let results = ::std::sync::Arc::into_inner(indices).unwrap().into_inner().unwrap();
                let mut indices = (0 .. container.len()).collect::<Vec<usize>>();
                indices.sort_by(|v1, v2| f64::total_cmp(&results[*v2], &results[*v1]));
                if reverse{
                    self.best_estimator = Some(container[indices[container.len() - 1]].clone());
                } else{
                    self.best_estimator = Some(container[indices[0]].clone());
                }
                let results = zip(container, results).map(|(model, score)| (model.to_string(), score)).collect::<Vec<(String, f64)>>();
                self.results = Some(results);
            }
        }
    };

    res.into()
}

// #[proc_macro_derive(CrossValidatorTwo, attributes(expand, validate))]
// pub fn derive_two(input: TokenStream) -> TokenStream {
//     let struct_input = parse_macro_input!(input as DeriveInput);
//     let name = &struct_input.ident;
//     let validator_name = format!("{}Validator", name);
//     let original_fields = if let syn::Data::Struct(syn::DataStruct {
//         fields: syn::Fields::Named(syn::FieldsNamed { ref named, .. }),
//         ..
//     }) = struct_input.data
//     {
//         named
//     } else {
//         unimplemented!()
//     };
//     let estimator = if let Some(est) = original_fields.iter().find(|&f| {
//         f.attrs.iter().any(|f| {
//             if *f.path().get_ident().unwrap() == Ident::new("expand", struct_input.span()){
//                 true
//             } else {
//                 false
//             }
//         })
//     }) {
//         est
//     } else {
//         panic!("No field is being expanded for validation")
//     };
//     let estimator_fields
//     quote! {}.into()
// }
