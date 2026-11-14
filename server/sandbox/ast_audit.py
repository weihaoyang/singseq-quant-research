"""
server/sandbox/ast_audit.py
--------------------------------------------------
AST 静态安全与未来函数深度审计器 (AI-Friendly AST Auditor)
（规范位置；旧 server/ast_audit.py 垫片已按 §3.8 移除）

功能：
1. 拦截高危系统调用与越权行为 (os, sys, socket, subprocess, eval, exec, __import__ 等)
2. 拦截一切文件读写 (open / Path.read_text / DataFrame.to_csv / pd.read_csv ...)
3. 拦截沙箱逃逸常见的 dunder 反射链 (().__class__.__subclasses__() 等)
4. 扫描时序未来函数 (Anti-Lookahead Bias):
   - 检测 Polars/Pandas 负向位移: .shift(-1), .shift(periods=-k), .lead(-k)
   - 检测 Qlib 负向引用: Ref(..., -k)
   - 检测直接引用预测标签列 (target_ret / fwd_ret ...) 的未来函数模式
5. 每条诊断都带 **节点类型 (node_type) + 行号 + 代码切片 + 修复建议**，
   使反作弊取证中心 (forensics hub) 能直接展示「为什么被拒」。

**注意**：本模块只做静态审查，不执行任何用户代码；主进程唯一允许
exec() 用户代码的地方是 server/services/factor_service.py::run_preflight。
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

__all__ = [
    "DiagnosticItem",
    "AuditResult",
    "FactorASTAuditor",
    "audit_factor_code",
]


@dataclass
class DiagnosticItem:
    severity: str             # "CRITICAL", "WARNING", "INFO"
    error_code: str           # "LOOKAHEAD_BIAS", "UNSAFE_IMPORT", "MISSING_CONTRACT", ...
    line: int
    column: int
    code_snippet: str
    explanation: str
    actionable_suggestion: str
    node_type: str = ""       # 被拦截的 AST 节点类型 (Import/Call/Attribute/Subscript/...)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditResult:
    passed: bool
    diagnostics: List[DiagnosticItem]
    factor_classes: List[str]
    model_classes: List[str]

    @property
    def criticals(self) -> List[DiagnosticItem]:
        return [d for d in self.diagnostics if d.severity == "CRITICAL"]

    @property
    def rejections(self) -> List[str]:
        """结构化的拒绝原因串（非 traceback），供 API 层直接返回给客户端。"""
        return [
            f"AST_REJECTED[{d.error_code}]@L{d.line}: {d.explanation}"
            for d in self.criticals
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "diagnostics": [asdict(d) for d in self.diagnostics],
            "factor_classes": self.factor_classes,
            "model_classes": self.model_classes,
        }


class FactorASTAuditor(ast.NodeVisitor):
    """对用户提交的因子源码做静态安全 / 未来函数审查。"""

    # ---- 禁止导入的模块（进程、文件系统、网络、反射序列化）----
    FORBIDDEN_MODULES = {
        # 进程 / 系统 / 文件系统
        "os", "sys", "subprocess", "shutil", "pathlib", "tempfile", "glob", "io",
        "ctypes", "importlib", "multiprocessing", "threading", "signal", "resource",
        "platform", "pty", "selectors", "queue", "site", "builtins",
        # 网络
        "socket", "ssl", "urllib", "http", "requests", "httpx", "telnetlib",
        "ftplib", "paramiko", "websockets", "aiohttp", "smtplib",
        # 反射 / 序列化 / 持久化
        "pickle", "shelve", "dbm", "marshal", "sqlite3", "winreg",
    }

    # ---- 禁止调用的内置函数（exec 仅沙箱子进程入口可用，用户代码一律禁止）----
    FORBIDDEN_CALLS = {
        "eval", "exec", "compile", "__import__", "open", "input",
        "breakpoint", "exit", "quit",
        # 反射内省：即使绕过 import 拦截，也不允许用 getattr 拼出 __globals__/
        # __subclasses__ 等逃逸链。
        "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir",
    }

    # ---- 禁止调用的危险方法：系统命令执行 / 进程生成（os.system、os.popen ...）----
    # 即使调用方通过别名绕过 `import os` 拦截，这一层也能点名拒绝。
    DANGEROUS_METHOD_CALLS = {
        "system", "popen", "fork", "forkpty", "startfile",
        "execl", "execle", "execlp", "execv", "execve", "execvp",
        "spawnl", "spawnv", "spawnlp", "spawnvp",
    }

    # ---- 禁止调用的方法：文件读写 / 外部数据载入 / 落盘 ----
    FORBIDDEN_METHODS = {
        # 文件系统
        "read_text", "write_text", "read_bytes", "write_bytes", "unlink",
        "rmdir", "mkdir", "makedirs", "rename", "replace", "touch", "chmod",
        "iterdir", "scandir", "listdir", "removedirs",
        # pandas / polars 外部数据 IO
        "read_csv", "read_table", "read_excel", "read_pickle", "read_parquet",
        "read_hdf", "read_sql", "read_sql_query", "read_fwf", "read_json",
        "scan_csv", "scan_parquet", "read_ipc", "read_database",
        # numpy file-backed arrays / binary IO (``np`` is otherwise allowed
        # for numeric factor math, so these must be denied explicitly).
        "load", "loadtxt", "genfromtxt", "memmap", "fromfile", "frombuffer",
        "open_memmap", "save", "savetxt", "savez", "savez_compressed", "tofile",
        # 落盘 / 剪贴板
        "to_csv", "to_pickle", "to_parquet", "to_excel", "to_hdf", "to_sql",
        "to_feather", "to_stata", "to_clipboard",
    }

    # ---- 禁止访问的属性名（模块逃逸链）----
    # `import os` 会被拦，但**属性链**仍能取到真实模块对象：`pd.io.common.os` 无需
    # 任何子模块导入即可拿到真正的 `os`（其 `os.environ` 含服务进程导出的 .env）。
    # 单纯拦 import 无法覆盖这条路，因此在取值形态上直接拒绝这些属性名。
    FORBIDDEN_ATTRIBUTES = {
        "os", "sys", "subprocess", "builtins", "importlib", "ctypes", "shutil",
        "pathlib", "socket", "winreg", "environ", "getenv", "putenv",
        "system", "popen", "spawnv", "execv", "fork", "startfile",
    }

    # ---- 未来函数：禁止直接引用预测标签列 ----
    LOOKAHEAD_COLUMNS = {
        "target_ret", "fwd_ret", "fwd_return", "fwd_ret_1d", "forward_return",
        "next_close", "fwd_close", "ret_fwd", "future_ret",
    }

    # ---- 负向位移方法：.shift(-k) / .lead(-k) ----
    NEGATIVE_SHIFT_METHODS = ("shift", "lead")

    # ---- dunder 反射：沙箱逃逸的主要通道（().__class__.__subclasses__()）----
    # 只点名**反射 / 逃逸**用 dunder。协议类 dunder（__init__ / __len__ /
    # __call__ / __eq__ ...）是因子类的正常写法 —— 官方模板就用
    # ``super().__init__(metadata)`` —— 一律放行，否则会误杀合法因子。
    DANGEROUS_DUNDERS = {
        # 继承链 / 类型跃迁
        "__class__", "__bases__", "__mro__", "__subclasses__",
        # 函数与模块内省
        "__globals__", "__code__", "__closure__", "__defaults__",
        "__kwdefaults__", "__dict__", "__builtins__", "__import__",
        "__loader__", "__self__", "__func__",
        # 属性劫持与反序列化执行
        "__getattribute__", "__getattr__", "__setattr__", "__delattr__",
        "__reduce__", "__reduce_ex__", "__deepcopy__",
        # 栈帧 / 异常对象
        "__traceback__", "__frame__", "__instancecheck__",
    }

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.diagnostics: List[DiagnosticItem] = []
        self.factor_classes: List[str] = []
        self.model_classes: List[str] = []

    # ------------------------------------------------------------------ utils
    def _get_line_snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()
        return ""

    def _reject(self, *, error_code: str, node: ast.AST, explanation: str,
                suggestion: str, severity: str = "CRITICAL") -> None:
        self.diagnostics.append(DiagnosticItem(
            severity=severity,
            error_code=error_code,
            line=getattr(node, "lineno", 0),
            column=getattr(node, "col_offset", 0),
            code_snippet=self._get_line_snippet(getattr(node, "lineno", 0)),
            explanation=explanation,
            actionable_suggestion=suggestion,
            node_type=type(node).__name__,
        ))

    @staticmethod
    def _is_negative_number(node: ast.AST) -> bool:
        """判断 AST 节点是否为显式负数 (如 -1, -5)。"""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(
                    node.operand.value, (int, float)):
                return node.operand.value > 0
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value < 0
        return False

    # ---------------------------------------------------------------- imports
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in self.FORBIDDEN_MODULES:
                self._reject(
                    error_code="UNSAFE_IMPORT", node=node,
                    explanation=f"禁止导入高危系统模块 `{alias.name}`，以保障服务器安全。",
                    suggestion="请仅使用 numpy, pandas, polars, scipy, torch 等量化计算库。",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in self.FORBIDDEN_MODULES:
                self._reject(
                    error_code="UNSAFE_IMPORT", node=node,
                    explanation=f"禁止从高危系统模块 `{node.module}` 导入对象。",
                    suggestion="请移除该导入，因子计算逻辑必须自包含于向量化运算中。",
                )
        self.generic_visit(node)

    # ------------------------------------------------------------------ calls
    def visit_Call(self, node: ast.Call):
        # 1. 禁止的内置函数 (eval / exec / __import__ / open ...)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.FORBIDDEN_CALLS:
                self._reject(
                    error_code="UNSAFE_CALL", node=node,
                    explanation=f"禁止调用敏感内置函数 `{func_name}()`。",
                    suggestion="因子算式必须使用静态纯函数或向量化表达式实现；"
                               "exec/eval 仅允许出现在沙箱子进程入口。",
                )
            # Qlib Ref(..., 负数) 未来函数
            if func_name == "Ref" and len(node.args) >= 2:
                if self._is_negative_number(node.args[1]):
                    self._reject(
                        error_code="LOOKAHEAD_BIAS", node=node,
                        explanation="检测到 Qlib Ref(..., 负数) 引用未来行情，"
                                    "存在严重前向偷窥偏误 (Lookahead Bias)！",
                        suggestion="因子特征必须只引用非负滞后周期 (如 Ref(close, 1))，"
                                   "预测目标仅由回测引擎计算。",
                    )

        # 2. 方法调用：文件 IO / 负向位移 / dunder 反射链
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr

            if attr_name in self.DANGEROUS_METHOD_CALLS:
                self._reject(
                    error_code="UNSAFE_CALL", node=node,
                    explanation=f"禁止调用 `{attr_name}()`：因子代码不得执行系统命令或生成进程。",
                    suggestion="因子只需对入参 df 做向量化计算；任何系统级调用都会被静态审计点名拦截。",
                )

            if attr_name in self.FORBIDDEN_METHODS:
                self._reject(
                    error_code="FILE_IO_DENIED", node=node,
                    explanation=f"禁止调用 `{attr_name}()`：因子不得进行任何文件读写或外部数据载入。",
                    suggestion="行情数据由沙箱面板统一注入，因子只需对入参 df 做向量化计算。",
                )

            if attr_name in self.NEGATIVE_SHIFT_METHODS:
                neg = False
                if node.args and self._is_negative_number(node.args[0]):
                    neg = True
                for kw in node.keywords:
                    if kw.arg in (None, "periods", "offset", "n") and \
                            self._is_negative_number(kw.value):
                        neg = True
                if neg:
                    self._reject(
                        error_code="LOOKAHEAD_BIAS", node=node,
                        explanation=f"检测到 `{attr_name}(负数)` 访问了前向未来数据，"
                                    "造成严重未来函数泄漏！",
                        suggestion="请使用正数做历史滞后 (如 .shift(1) / .shift(5))，"
                                   "严禁取未来收益率作为因子输入。",
                    )

            if attr_name.startswith("__") and attr_name.endswith("__") \
                    and attr_name in self.DANGEROUS_DUNDERS:
                self._reject(
                    error_code="UNSAFE_DUNDER", node=node,
                    explanation=f"禁止访问魔术属性 `{attr_name}`：dunder 反射链是"
                                "典型的沙箱逃逸通道 (如 __class__.__subclasses__)。",
                    suggestion="请移除反射链，只使用公开的 pandas / numpy / polars API。",
                )

        # 3. 传给函数的字符串常量 (pl.col("target_ret") / groupby("target_ret") ...)
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and arg.value in self.LOOKAHEAD_COLUMNS:
                self._reject(
                    error_code="LOOKAHEAD_COLUMN", node=node,
                    explanation=f"检测到引用预测标签列 `{arg.value}`：因子输入不得"
                                "直接读取未来收益标签。",
                    suggestion="标签/未来收益只由回测引擎在评估阶段计算，"
                               "因子只能依赖行情与财务截面数据。",
                )

        self.generic_visit(node)

    # ------------------------------------------------------------- subscripts
    def visit_Subscript(self, node: ast.Subscript):
        """df["target_ret"] 这类直接读取预测标签列的写法。"""
        # 禁止 ``__builtins__["__import__"]`` 这类下标取内置的逃逸写法
        # （visit_Call 只识别 Name/Attribute，下标形态必须在这里补拦）。
        if isinstance(node.value, ast.Name) and node.value.id == "__builtins__":
            self._reject(
                error_code="UNSAFE_SUBSCRIPT", node=node,
                explanation="禁止下标访问 `__builtins__`：典型的沙箱逃逸写法。",
                suggestion="请移除反射写法，只使用公开的 pandas / numpy / polars API。",
            )
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str) \
                and sl.value in self.LOOKAHEAD_COLUMNS:
            self._reject(
                error_code="LOOKAHEAD_COLUMN", node=node,
                explanation=f"检测到下标直接读取预测标签列 `{sl.value}`，存在未来函数风险。",
                suggestion="标签列由回测引擎持有，因子请使用行情/财务列"
                           "(open/high/low/close/vol/amount/adj_factor ...)。",
            )
        self.generic_visit(node)

    # ------------------------------------------------------------- attributes
    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__") and node.attr.endswith("__") \
                and node.attr in self.DANGEROUS_DUNDERS:
            # 取值形态：obj.__globals__ / ().__class__ / x.__dict__
            # （调用形态已由 visit_Call 覆盖，这里覆盖纯取值）
            self._reject(
                error_code="UNSAFE_DUNDER", node=node,
                explanation=f"禁止读取魔术属性 `{node.attr}`。",
                suggestion="请移除反射/内省写法，只使用公开 API。",
            )
        elif node.attr in self.FORBIDDEN_ATTRIBUTES:
            self._reject(
                error_code="UNSAFE_ATTRIBUTE", node=node,
                explanation=f"禁止访问属性 `.{node.attr}`：该属性链可绕到 os/sys 等系统模块。",
                suggestion="因子只能使用入参 df 与 numpy/pandas/polars 的公开数值算子；"
                           "不得访问解释器、进程或环境对象。",
            )
        self.generic_visit(node)

    # ---------------------------------------------------------------- classes
    def visit_ClassDef(self, node: ast.ClassDef):
        for base in node.bases:
            base_id = ""
            if isinstance(base, ast.Name):
                base_id = base.id
            elif isinstance(base, ast.Attribute):
                base_id = base.attr

            if base_id == "BaseFactor":
                self.factor_classes.append(node.name)
            elif base_id in ("Module", "nn.Module"):
                self.model_classes.append(node.name)

        self.generic_visit(node)


def audit_factor_code(source_code: str) -> AuditResult:
    """对传入的 Python 源码进行完整的静态语法树深度审计。

    返回的 :class:`AuditResult` 带 `passed` / `diagnostics`（含 node + line），
    供 factor_service、sandbox worker 与反作弊取证中心共用。
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return AuditResult(
            passed=False,
            diagnostics=[
                DiagnosticItem(
                    severity="CRITICAL",
                    error_code="SYNTAX_ERROR",
                    line=e.lineno or 1,
                    column=e.offset or 0,
                    code_snippet=(e.text or "").strip(),
                    explanation=f"Python 语法错误: {e.msg}",
                    actionable_suggestion="请修正代码中的拼写或语法错误后重新提交。",
                    node_type="SyntaxError",
                )
            ],
            factor_classes=[],
            model_classes=[],
        )

    auditor = FactorASTAuditor(source_code)
    auditor.visit(tree)

    # 结构完整性提示（非致命）：类式因子需继承 BaseFactor，函数式因子需提供 compute_factor
    if not auditor.factor_classes and not auditor.model_classes:
        has_functional_entry = any(
            isinstance(n, ast.FunctionDef) and n.name == "compute_factor"
            for n in ast.walk(tree)
        )
        if not has_functional_entry:
            auditor.diagnostics.append(DiagnosticItem(
                severity="WARNING",
                error_code="NO_TARGET_CLASS",
                line=1,
                column=0,
                code_snippet="",
                explanation="未检测到继承自 `BaseFactor` 的单因子类，也未检测到函数式入口 "
                            "`def compute_factor(df) -> pd.Series`。",
                actionable_suggestion="请参考 quant_client/templates/，实现 BaseFactor 子类或 "
                                      "`compute_factor(df) -> pd.Series` 入口。",
                node_type="Module",
            ))

    critical_errors = [d for d in auditor.diagnostics if d.severity == "CRITICAL"]

    # 去重：Call 分支与 Attribute/Subscript 分支可能对同一节点各报一条
    seen: set[tuple] = set()
    unique_diagnostics: List[DiagnosticItem] = []
    for d in auditor.diagnostics:
        key = (d.error_code, d.line, d.node_type, d.code_snippet)
        if key in seen:
            continue
        seen.add(key)
        unique_diagnostics.append(d)

    return AuditResult(
        passed=len(critical_errors) == 0,
        diagnostics=unique_diagnostics,
        factor_classes=auditor.factor_classes,
        model_classes=auditor.model_classes,
    )
