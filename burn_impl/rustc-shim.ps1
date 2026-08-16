# rustc 透传 wrapper（绕过全局 cargo 配置的 sccache，且无 cmd.exe 8191 字符限制）。
# 用法（cargo config）：rustc-wrapper = "powershell -NoProfile -ExecutionPolicy Bypass -File <本脚本>"
# 第一个参数是真实 rustc 路径，其余原样转发。
$rustc = $args[0]
$rest = @($args[1..($args.Length - 1)])
& $rustc @rest
exit $LASTEXITCODE
