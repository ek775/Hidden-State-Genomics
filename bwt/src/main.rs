fn main() {
    eprintln!(
        "bwt is a Python extension module, not a standalone binary.\n\
         Build it with:\n\
             cd bwt && maturin develop --release\n\
         Then import from Python:\n\
             from bwt import BwtIndex"
    );
    std::process::exit(1);
}
