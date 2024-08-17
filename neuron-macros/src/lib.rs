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
            let value = val.base10_parse::<i64>().unwrap();
            if value < 0 {
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
    let better = fields
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
    let into_validator_iterator = fields
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
    let res = quote! {

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
            pub fn fit(dataset: ::neuron::base_array::BaseDataset, target: &str, evaluation: fn(::ndarray::Array1<f64>, ::ndarray::Array1<f64>) -> std::vec::Vec<f64>, reverse: bool){
                //so basically we want to make multiple copies of the estimators, which means another traipsing of the fields


            }
        }
    };

    res.into()
}
