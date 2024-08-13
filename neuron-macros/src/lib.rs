use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Lit};

#[proc_macro]
pub fn dtype(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Lit);
    let dtype = match input {
        Lit::Str(val) => {
            quote! {
                ::neuron::DType::Object(Box::new(#val.to_string()))
            }
        }
        Lit::ByteStr(val) => {
            quote! {
                ::neuron::DType::Object(Box::new(::std::string::String::from_utf8(#val)))
            }
        }
        Lit::Char(val) => {
            quote! {
                ::neuron::DType::Object(Box::new(::std::string::String::from(#val)))
            }
        }
        Lit::Int(val) => {
            //find if it's negative, if it is, make it float, else unsigned int
            let value = val.base10_parse::<i64>().unwrap();
            if value < 0 {
                quote! {
                    ::neuron::DType::F64(#value as f64)
                }
            } else {
                quote! {
                    ::neuron::DType::U64(#value as u64)
                }
            }
        }
        Lit::Float(val) => {
            //remember, the float CANNOT be NaN or weird in any manner shape or form
            let val = val.base10_parse::<f64>().unwrap();
            if val.is_normal() {
                quote! {
                    ::neuron::dtype::DType::F64(val)
                }
            } else {
                quote! {
                    ::neuron::dtype::DType::None
                }
            }
        }
        Lit::Bool(val) => {
            quote! {
                ::neuron::dtype::DType::U32(#val.value() as u32)
            }
        }
        _ => quote! {::neuron::dtype::DType::None},
    };
    dtype.into()
}

#[proc_macro]
pub fn internal_dtype(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Lit);
    let dtype = match input {
        Lit::Str(val) => {
            quote! {
                DType::Object(Box::new(#val.to_string()))
            }
        }
        Lit::ByteStr(val) => {
            quote! {
                DType::Object(Box::new(::std::string::String::from_utf8(#val)))
            }
        }
        Lit::Char(val) => {
            quote! {
                DType::Object(Box::new(::std::string::String::from(#val)))
            }
        }
        Lit::Int(val) => {
            //find if it's negative, if it is, make it float, else unsigned int
            let value = val.base10_parse::<i64>().unwrap();
            if value < 0 {
                quote! {
                    DType::F64(#value as f64)
                }
            } else {
                quote! {
                    DType::U64(#value as u64)
                }
            }
        }
        Lit::Float(val) => {
            //remember, the float CANNOT be NaN or weird in any manner shape or form
            let val = val.base10_parse::<f64>().unwrap();
            if val.is_normal() {
                quote! {
                    DType::F64(#val)
                }
            } else {
                quote! {
                    DType::None
                }
            }
        }
        Lit::Bool(val) => {
            quote! {
                DType::U32(#val.value() as u32)
            }
        }
        _ => quote! {DType::None},
    };
    dtype.into()
}
