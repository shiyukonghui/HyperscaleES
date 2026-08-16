// rustc 透传 wrapper（绕过全局 cargo 配置的 sccache；无 cmd.exe 8191 字符限制）。
// 编译：rustc -O rustc-shim.rs -o rustc-shim.exe
// 用法（cargo config）：rustc-wrapper = "F:\\...\\rustc-shim.exe"
// 第一个参数是真实 rustc 路径，其余原样转发，退出码透传。
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        std::process::exit(2);
    }
    let status = Command::new(&args[0]).args(&args[1..]).status();
    match status {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("rustc-shim: 启动 {} 失败: {e}", args[0]);
            std::process::exit(2);
        }
    }
}
