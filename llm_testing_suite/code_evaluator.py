"""
Code Evaluation Module for LLM Testing Suite
Provides comprehensive code evaluation across multiple programming languages.
"""

import os
import re
import ast
import logging
import subprocess
import tempfile
from sentence_transformers import util

logger = logging.getLogger(__name__)


class CodeEvaluator:
    """
    Handles all code-specific evaluation metrics including syntax validation,
    execution testing, quality metrics, security scanning, and semantic analysis.
    """
    
    def __init__(self, embedder=None, save_json_callback=None, display_table_callback=None):
        """
        Initialize the CodeEvaluator.
        
        Args:
            embedder: Sentence transformer model for semantic analysis
            save_json_callback: Callback function to save results as JSON
            display_table_callback: Callback function to display results as table
        """
        self.embedder = embedder
        self.save_json = save_json_callback
        self.display_table = display_table_callback
    
    def code_syntax_validity(self, code_response, language="python", save_json=False, return_type="dict"):
        """
        Check if generated code is syntactically valid.
        Supports: python, javascript, java, cpp, go, rust, ruby, php, typescript
        """
        is_valid = False
        error_message = None
        
        try:
            if language.lower() == "python":
                # Python: Use AST parser for comprehensive syntax checking
                ast.parse(code_response)
                is_valid = True
                
            elif language.lower() in ["javascript", "js"]:
                # JavaScript: Use Node.js syntax checker
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['node', '--check', f.name], 
                                              capture_output=True, text=True, timeout=5)
                        is_valid = result.returncode == 0
                        error_message = result.stderr if not is_valid else None
                    os.unlink(f.name)
                except FileNotFoundError:
                    # Node.js not available, fall back to basic check
                    is_valid = 'function' in code_response or 'const' in code_response or 'let' in code_response or 'var' in code_response
                    if not is_valid:
                        error_message = "Node.js not available for validation; basic check failed"
                        
            elif language.lower() in ["typescript", "ts"]:
                # TypeScript: Use tsc for syntax checking
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['tsc', '--noEmit', f.name], 
                                              capture_output=True, text=True, timeout=5)
                        is_valid = result.returncode == 0
                        error_message = result.stderr if not is_valid else None
                    os.unlink(f.name)
                except FileNotFoundError:
                    # TypeScript compiler not available, fall back to basic check
                    is_valid = 'function' in code_response or 'const' in code_response or 'interface' in code_response or 'type' in code_response
                    if not is_valid:
                        error_message = "TypeScript compiler not available; basic check failed"
                        
            elif language.lower() == "java":
                # Java: Use javac for syntax checking
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                        # Extract class name from code if possible
                        class_match = re.search(r'public\s+class\s+(\w+)', code_response)
                        if class_match:
                            class_name = class_match.group(1)
                            java_file = f.name.replace('.java', f'_{class_name}.java')
                            with open(java_file, 'w') as jf:
                                jf.write(code_response)
                            result = subprocess.run(['javac', '-Xlint:none', java_file], 
                                                  capture_output=True, text=True, timeout=5)
                            is_valid = result.returncode == 0
                            error_message = result.stderr if not is_valid else None
                            os.unlink(java_file)
                        else:
                            f.write(code_response)
                            f.flush()
                            result = subprocess.run(['javac', '-Xlint:none', f.name], 
                                                  capture_output=True, text=True, timeout=5)
                            is_valid = result.returncode == 0
                            error_message = result.stderr if not is_valid else None
                        os.unlink(f.name)
                except FileNotFoundError:
                    # javac not available, fall back to basic check
                    is_valid = ('class ' in code_response or 'interface ' in code_response) and '{' in code_response and '}' in code_response
                    if not is_valid:
                        error_message = "javac not available; basic check failed"
                        
            elif language.lower() in ["cpp", "c++"]:
                # C++: Use g++ for syntax checking
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['g++', '-fsyntax-only', f.name], 
                                              capture_output=True, text=True, timeout=5)
                        is_valid = result.returncode == 0
                        error_message = result.stderr if not is_valid else None
                    os.unlink(f.name)
                except FileNotFoundError:
                    # g++ not available, fall back to basic check
                    is_valid = '#include' in code_response or 'using namespace' in code_response or 'int main' in code_response
                    if not is_valid:
                        error_message = "g++ not available; basic check failed"
                        
            elif language.lower() == "c":
                # C: Use gcc for syntax checking
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['gcc', '-fsyntax-only', f.name], 
                                              capture_output=True, text=True, timeout=5)
                        is_valid = result.returncode == 0
                        error_message = result.stderr if not is_valid else None
                    os.unlink(f.name)
                except FileNotFoundError:
                    # gcc not available, fall back to basic check
                    is_valid = '#include' in code_response or 'int main' in code_response
                    if not is_valid:
                        error_message = "gcc not available; basic check failed"
                        
            elif language.lower() == "go":
                # Go: Use go compiler for syntax checking
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['go', 'fmt', f.name], 
                                              capture_output=True, text=True, timeout=5)
                        is_valid = result.returncode == 0
                        error_message = result.stderr if not is_valid else None
                    os.unlink(f.name)
                except FileNotFoundError:
                    # go not available, fall back to basic check
                    is_valid = 'package ' in code_response and 'func ' in code_response
                    if not is_valid:
                        error_message = "go not available; basic check failed"
                        
            elif language.lower() == "rust":
                # Rust: Use rustc for syntax checking
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['rustc', '--crate-type', 'lib', '-Z', 'parse-only', f.name], 
                                              capture_output=True, text=True, timeout=5)
                        is_valid = result.returncode == 0
                        error_message = result.stderr if not is_valid else None
                    os.unlink(f.name)
                except FileNotFoundError:
                    # rustc not available, fall back to basic check
                    is_valid = 'fn ' in code_response or 'use ' in code_response
                    if not is_valid:
                        error_message = "rustc not available; basic check failed"
                        
            elif language.lower() == "ruby":
                # Ruby: Use ruby -c for syntax checking
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['ruby', '-c', f.name], 
                                              capture_output=True, text=True, timeout=5)
                        is_valid = result.returncode == 0
                        error_message = result.stderr if not is_valid else None
                    os.unlink(f.name)
                except FileNotFoundError:
                    # ruby not available, fall back to basic check
                    is_valid = 'def ' in code_response or 'class ' in code_response or 'module ' in code_response
                    if not is_valid:
                        error_message = "ruby not available; basic check failed"
                        
            elif language.lower() == "php":
                # PHP: Use php -l for syntax checking
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.php', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['php', '-l', f.name], 
                                              capture_output=True, text=True, timeout=5)
                        is_valid = result.returncode == 0
                        error_message = result.stdout if not is_valid else None
                    os.unlink(f.name)
                except FileNotFoundError:
                    # php not available, fall back to basic check
                    is_valid = '<?php' in code_response or 'function ' in code_response
                    if not is_valid:
                        error_message = "php not available; basic check failed"
            else:
                error_message = f"Unsupported language: {language}"
                    
        except SyntaxError as e:
            error_message = str(e)
        except Exception as e:
            error_message = f"Validation error: {str(e)}"
        
        result = {
            "code": code_response,
            "language": language,
            "syntax_valid": is_valid,
            "error": error_message
        }
        
        if save_json and self.save_json:
            self.save_json(result, test_name="code_syntax_validity")
        if return_type in ["table", "both"] and self.display_table:
            self.display_table(result, title="Code Syntax Validity")
        
        logger.info(f"Syntax validity for {language}: {is_valid}")
        return result

    def code_execution_test(self, code_response, test_cases, language="python", 
                           timeout=5, save_json=False, return_type="dict"):
        """
        Execute code with test cases and verify outputs.
        test_cases: list of dicts with 'input' and 'expected_output'
        Supports: python, javascript, java, cpp, c, go, ruby, php
        """
        passed_tests = 0
        total_tests = len(test_cases)
        test_results = []
        
        for idx, test in enumerate(test_cases):
            try:
                lang = language.lower()
                test_input = test.get('input', '')
                expected_output = str(test['expected_output']).strip()
                
                if lang == "python":
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['python', f.name], 
                                              input=test_input,
                                              capture_output=True, 
                                              text=True, 
                                              timeout=timeout)
                    os.unlink(f.name)
                    
                elif lang in ["javascript", "js"]:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['node', f.name], 
                                              input=test_input,
                                              capture_output=True, 
                                              text=True, 
                                              timeout=timeout)
                    os.unlink(f.name)
                    
                elif lang == "java":
                    # Extract class name
                    class_match = re.search(r'public\s+class\s+(\w+)', code_response)
                    if class_match:
                        class_name = class_match.group(1)
                        java_file = f'{class_name}.java'
                        with open(java_file, 'w') as f:
                            f.write(code_response)
                        # Compile
                        compile_result = subprocess.run(['javac', java_file], 
                                                      capture_output=True, text=True, timeout=timeout)
                        if compile_result.returncode == 0:
                            # Run
                            result = subprocess.run(['java', class_name], 
                                                  input=test_input,
                                                  capture_output=True, 
                                                  text=True, 
                                                  timeout=timeout)
                        else:
                            result = compile_result
                        # Cleanup
                        os.unlink(java_file)
                        if os.path.exists(f'{class_name}.class'):
                            os.unlink(f'{class_name}.class')
                    else:
                        raise Exception("No public class found in Java code")
                        
                elif lang in ["cpp", "c++"]:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        exe_file = f.name + '.out'
                        # Compile
                        compile_result = subprocess.run(['g++', f.name, '-o', exe_file], 
                                                      capture_output=True, text=True, timeout=timeout)
                        if compile_result.returncode == 0:
                            # Run
                            result = subprocess.run([exe_file], 
                                                  input=test_input,
                                                  capture_output=True, 
                                                  text=True, 
                                                  timeout=timeout)
                            os.unlink(exe_file)
                        else:
                            result = compile_result
                    os.unlink(f.name)
                    
                elif lang == "c":
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        exe_file = f.name + '.out'
                        # Compile
                        compile_result = subprocess.run(['gcc', f.name, '-o', exe_file], 
                                                      capture_output=True, text=True, timeout=timeout)
                        if compile_result.returncode == 0:
                            # Run
                            result = subprocess.run([exe_file], 
                                                  input=test_input,
                                                  capture_output=True, 
                                                  text=True, 
                                                  timeout=timeout)
                            os.unlink(exe_file)
                        else:
                            result = compile_result
                    os.unlink(f.name)
                    
                elif lang == "go":
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['go', 'run', f.name], 
                                              input=test_input,
                                              capture_output=True, 
                                              text=True, 
                                              timeout=timeout)
                    os.unlink(f.name)
                    
                elif lang == "ruby":
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['ruby', f.name], 
                                              input=test_input,
                                              capture_output=True, 
                                              text=True, 
                                              timeout=timeout)
                    os.unlink(f.name)
                    
                elif lang == "php":
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.php', delete=False) as f:
                        f.write(code_response)
                        f.flush()
                        result = subprocess.run(['php', f.name], 
                                              input=test_input,
                                              capture_output=True, 
                                              text=True, 
                                              timeout=timeout)
                    os.unlink(f.name)
                else:
                    raise Exception(f"Unsupported language for execution: {language}")
                
                actual_output = result.stdout.strip()
                passed = actual_output == expected_output
                if passed:
                    passed_tests += 1
                
                test_results.append({
                    "test_id": idx,
                    "input": test_input,
                    "expected": expected_output,
                    "actual": actual_output,
                    "passed": passed,
                    "error": result.stderr if result.returncode != 0 else None
                })
                    
            except subprocess.TimeoutExpired:
                test_results.append({
                    "test_id": idx,
                    "passed": False,
                    "error": "Execution timeout"
                })
            except Exception as e:
                test_results.append({
                    "test_id": idx,
                    "passed": False,
                    "error": str(e)
                })
        
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        result = {
            "code": code_response,
            "pass_rate": pass_rate,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "test_results": test_results
        }
        
        if save_json and self.save_json:
            self.save_json(result, test_name="code_execution_test")
        if return_type in ["table", "both"] and self.display_table:
            self.display_table(result, title="Code Execution Test")
        
        logger.info(f"Code execution pass rate: {pass_rate:.2%}")
        return result

    def code_quality_metrics(self, code_response, language="python", 
                            save_json=False, return_type="dict"):
        """
        Analyze code quality metrics: complexity, documentation, structure.
        Supports: python, javascript, java, cpp, c, go, ruby, php, rust, typescript
        """
        lang = language.lower()
        
        # Basic metrics applicable to all languages
        metrics = {
            "lines_of_code": len(code_response.split('\n')),
            "blank_lines": code_response.count('\n\n'),
        }
        
        # Language-specific comment detection
        if lang == "python":
            metrics["has_comments"] = '#' in code_response
            metrics["has_docstring"] = '"""' in code_response or "'''" in code_response
        elif lang in ["javascript", "js", "typescript", "ts", "java", "cpp", "c++", "c", "go", "rust", "php"]:
            metrics["has_comments"] = '//' in code_response or '/*' in code_response
            metrics["has_docstring"] = '/**' in code_response  # JSDoc/JavaDoc style
        elif lang == "ruby":
            metrics["has_comments"] = '#' in code_response
            metrics["has_docstring"] = '=begin' in code_response
        else:
            metrics["has_comments"] = '#' in code_response or '//' in code_response or '/*' in code_response
            metrics["has_docstring"] = False
        
        # Language-specific detailed analysis
        if lang == "python":
            try:
                tree = ast.parse(code_response)
                metrics["num_functions"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                metrics["num_classes"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
                
                # Cyclomatic complexity
                complexity = sum(1 for node in ast.walk(tree) 
                               if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)))
                metrics["cyclomatic_complexity"] = complexity
                metrics["has_error_handling"] = any(isinstance(node, ast.Try) for node in ast.walk(tree))
            except:
                pass
                
        elif lang in ["javascript", "js", "typescript", "ts"]:
            # Basic pattern matching for JS/TS
            metrics["num_functions"] = len(re.findall(r'\bfunction\s+\w+|const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|\w+\s*:\s*(?:async\s*)?\([^)]*\)\s*=>', code_response))
            metrics["num_classes"] = len(re.findall(r'\bclass\s+\w+', code_response))
            metrics["has_error_handling"] = 'try' in code_response and 'catch' in code_response
            # Simplified complexity
            complexity = len(re.findall(r'\b(if|while|for|case)\b', code_response))
            metrics["cyclomatic_complexity"] = complexity
            
        elif lang == "java":
            metrics["num_functions"] = len(re.findall(r'(?:public|private|protected|static|\s) +[\w<>\[\]]+\s+(\w+) *\([^\)]*\) *\{', code_response))
            metrics["num_classes"] = len(re.findall(r'\b(?:public|private)?\s*(?:abstract|final)?\s*class\s+\w+', code_response))
            metrics["has_error_handling"] = 'try' in code_response and 'catch' in code_response
            complexity = len(re.findall(r'\b(if|while|for|case)\b', code_response))
            metrics["cyclomatic_complexity"] = complexity
            
        elif lang in ["cpp", "c++", "c"]:
            metrics["num_functions"] = len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*\{', code_response))
            metrics["num_classes"] = len(re.findall(r'\bclass\s+\w+', code_response))
            metrics["has_error_handling"] = 'try' in code_response and 'catch' in code_response
            complexity = len(re.findall(r'\b(if|while|for|case)\b', code_response))
            metrics["cyclomatic_complexity"] = complexity
            
        elif lang == "go":
            metrics["num_functions"] = len(re.findall(r'\bfunc\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+\s*\(', code_response))
            metrics["num_classes"] = len(re.findall(r'\btype\s+\w+\s+struct', code_response))  # structs in Go
            metrics["has_error_handling"] = 'if err != nil' in code_response or 'panic' in code_response
            complexity = len(re.findall(r'\b(if|for|case|select)\b', code_response))
            metrics["cyclomatic_complexity"] = complexity
            
        elif lang == "ruby":
            metrics["num_functions"] = len(re.findall(r'\bdef\s+\w+', code_response))
            metrics["num_classes"] = len(re.findall(r'\bclass\s+\w+', code_response))
            metrics["has_error_handling"] = 'rescue' in code_response or 'begin' in code_response
            complexity = len(re.findall(r'\b(if|while|for|case)\b', code_response))
            metrics["cyclomatic_complexity"] = complexity
            
        elif lang == "php":
            metrics["num_functions"] = len(re.findall(r'\bfunction\s+\w+', code_response))
            metrics["num_classes"] = len(re.findall(r'\bclass\s+\w+', code_response))
            metrics["has_error_handling"] = 'try' in code_response and 'catch' in code_response
            complexity = len(re.findall(r'\b(if|while|for|case)\b', code_response))
            metrics["cyclomatic_complexity"] = complexity
            
        elif lang == "rust":
            metrics["num_functions"] = len(re.findall(r'\bfn\s+\w+', code_response))
            metrics["num_classes"] = len(re.findall(r'\bstruct\s+\w+', code_response)) + len(re.findall(r'\benum\s+\w+', code_response))
            metrics["has_error_handling"] = 'Result<' in code_response or 'Option<' in code_response or 'match' in code_response
            complexity = len(re.findall(r'\b(if|while|for|match)\b', code_response))
            metrics["cyclomatic_complexity"] = complexity
        
        # Calculate quality score (0-100)
        quality_score = 0
        if metrics.get("has_comments"): quality_score += 20
        if metrics.get("has_docstring"): quality_score += 20
        if metrics.get("has_error_handling", False): quality_score += 20
        if metrics.get("cyclomatic_complexity", 0) < 10: quality_score += 20
        if metrics.get("num_functions", 0) > 0: quality_score += 20
        
        result = {
            "code": code_response,
            "language": language,
            "quality_score": quality_score,
            "metrics": metrics
        }
        
        if save_json and self.save_json:
            self.save_json(result, test_name="code_quality_metrics")
        if return_type in ["table", "both"] and self.display_table:
            self.display_table(result, title="Code Quality Metrics")
        
        logger.info(f"Code quality score: {quality_score}/100")
        return result

    def code_security_scan(self, code_response, language="python", 
                          save_json=False, return_type="dict"):
        """
        Scan code for common security vulnerabilities and anti-patterns.
        Supports: python, javascript, java, cpp, c, go, ruby, php, rust, typescript
        """
        vulnerabilities = []
        lang = language.lower()
        
        # Language-specific security patterns
        if lang == "python":
            security_patterns = {
                "SQL Injection": [r"execute\s*\(.*%.*\)", r"\.format\s*\(.*SELECT", r"cursor\.execute\s*\(.*\+"],
                "Command Injection": [r"os\.system\s*\(", r"subprocess\.call\s*\(.*shell=True", r"subprocess\.Popen\s*\(.*shell=True"],
                "Hardcoded Secrets": [r"password\s*=\s*['\"](?!.*\{)[\w@#$%^&*]+['\"]", r"api_key\s*=\s*['\"][\w-]+['\"]", r"secret\s*=\s*['\"][\w-]+['\"]"],
                "Unsafe Deserialization": [r"pickle\.loads", r"yaml\.load\s*\((?!.*Loader=)", r"eval\s*\(", r"exec\s*\("],
                "Path Traversal": [r"\.\./", r"os\.path\.join\s*\(.*\.\."],
                "Unsafe Input": [r"input\s*\((?!.*(?:int|float|str)\s*\()"],
            }
            
        elif lang in ["javascript", "js", "typescript", "ts"]:
            security_patterns = {
                "SQL Injection": [r"query\s*\(.*\+", r"execute\s*\(.*\+", r"\$\{.*\}.*SELECT"],
                "Command Injection": [r"exec\s*\(", r"eval\s*\(", r"child_process.*exec"],
                "XSS": [r"innerHTML\s*=", r"document\.write\s*\(", r"\.html\s*\(.*\+"],
                "Hardcoded Secrets": [r"password\s*[:=]\s*['\"][\w@#$%^&*]+['\"]", r"apiKey\s*[:=]\s*['\"][\w-]+['\"]"],
                "Unsafe Deserialization": [r"JSON\.parse\s*\((?!.*try)", r"eval\s*\("],
                "Path Traversal": [r"\.\./", r"path\.join\s*\(.*\.\."],
            }
            
        elif lang == "java":
            security_patterns = {
                "SQL Injection": [r"Statement\.execute\s*\(.*\+", r"createStatement\s*\(\).*executeQuery\s*\(.*\+"],
                "Command Injection": [r"Runtime\.exec\s*\(", r"ProcessBuilder\s*\(.*\+"],
                "Hardcoded Secrets": [r"password\s*=\s*\"[\w@#$%^&*]+\"", r"apiKey\s*=\s*\"[\w-]+\""],
                "Unsafe Deserialization": [r"ObjectInputStream\s*\(", r"readObject\s*\("],
                "Path Traversal": [r"\.\./", r"File\s*\(.*\.\."],
                "XXE": [r"DocumentBuilderFactory(?!.*setFeature)"],
            }
            
        elif lang in ["cpp", "c++", "c"]:
            security_patterns = {
                "Buffer Overflow": [r"gets\s*\(", r"strcpy\s*\(", r"sprintf\s*\(", r"strcat\s*\("],
                "Command Injection": [r"system\s*\(", r"popen\s*\(", r"exec\w*\s*\("],
                "Format String": [r"printf\s*\((?!.*\")", r"sprintf\s*\((?!.*\")"],
                "Memory Leaks": [r"malloc\s*\((?!.*free)", r"new\s+\w+(?!.*delete)"],
                "Hardcoded Secrets": [r"password\s*=\s*\"[\w@#$%^&*]+\"", r"api_key\s*=\s*\"[\w-]+\""],
            }
            
        elif lang == "go":
            security_patterns = {
                "SQL Injection": [r"db\.Exec\s*\(.*\+", r"db\.Query\s*\(.*fmt\.Sprintf"],
                "Command Injection": [r"exec\.Command\s*\(.*\+", r"os\.system"],
                "Hardcoded Secrets": [r"password\s*:=\s*\"[\w@#$%^&*]+\"", r"apiKey\s*:=\s*\"[\w-]+\""],
                "Path Traversal": [r"\.\./", r"filepath\.Join\s*\(.*\.\."],
                "Unsafe Deserialization": [r"gob\.Decode", r"json\.Unmarshal(?!.*error)"],
            }
            
        elif lang == "php":
            security_patterns = {
                "SQL Injection": [r"mysql_query\s*\(.*\$", r"\->query\s*\(.*\$", r"mysqli_query\s*\(.*\$"],
                "Command Injection": [r"exec\s*\(", r"system\s*\(", r"passthru\s*\(", r"shell_exec\s*\(", r"`.*\$"],
                "XSS": [r"echo\s+\$(?!.*htmlspecialchars)", r"print\s+\$(?!.*htmlspecialchars)"],
                "File Inclusion": [r"include\s*\(?\s*\$", r"require\s*\(?\s*\$"],
                "Hardcoded Secrets": [r"\$password\s*=\s*['\"][\w@#$%^&*]+['\"]", r"\$api_key\s*=\s*['\"][\w-]+['\"]"],
                "Unsafe Deserialization": [r"unserialize\s*\("],
            }
            
        elif lang == "ruby":
            security_patterns = {
                "SQL Injection": [r"execute\s*\(.*#\{", r"find_by_sql\s*\(.*#\{"],
                "Command Injection": [r"system\s*\(", r"exec\s*\(", r"`.*#\{"],
                "Hardcoded Secrets": [r"password\s*=\s*['\"][\w@#$%^&*]+['\"]", r"api_key\s*=\s*['\"][\w-]+['\"]"],
                "Unsafe Deserialization": [r"Marshal\.load", r"YAML\.load(?!.*safe_load)"],
                "Path Traversal": [r"\.\./"],
            }
            
        elif lang == "rust":
            security_patterns = {
                "Command Injection": [r"Command::new\s*\(.*\+", r"std::process::Command"],
                "Hardcoded Secrets": [r"password\s*=\s*\"[\w@#$%^&*]+\"", r"api_key\s*=\s*\"[\w-]+\""],
                "Unsafe Code": [r"unsafe\s*\{"],
                "Path Traversal": [r"\.\./"],
            }
        else:
            # Generic patterns for unknown languages
            security_patterns = {
                "Hardcoded Secrets": [r"password\s*[=:]\s*['\"][\w@#$%^&*]+['\"]", r"api[_-]?key\s*[=:]\s*['\"][\w-]+['\"]"],
                "Path Traversal": [r"\.\./"],
                "Eval Usage": [r"\beval\s*\("],
            }
        
        # Check for vulnerabilities
        flagged_categories = set()  # Track unique vulnerability categories
        for vuln_type, patterns in security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_response, re.IGNORECASE):
                    vulnerabilities.append({
                        "type": vuln_type,
                        "pattern": pattern,
                        "severity": "HIGH" if vuln_type in ["SQL Injection", "Command Injection", "Buffer Overflow"] else "MEDIUM"
                    })
                    flagged_categories.add(vuln_type)  # Add category to set

        # Calculate SRS: |F| / |C| as per paper formula
        total_categories = len(security_patterns)
        flagged_count = len(flagged_categories)
        srs = flagged_count / total_categories if total_categories > 0 else 0.0

        is_secure = len(vulnerabilities) == 0
        
        result = {
            "code": code_response,
            "language": language,
            "is_secure": is_secure,
            "vulnerability_count": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "srs": srs,  # Add the SRS score
            "flagged_categories": flagged_count,
            "total_categories": total_categories
        }
        
        if save_json and self.save_json:
            self.save_json(result, test_name="code_security_scan")
        if return_type in ["table", "both"] and self.display_table:
            self.display_table(result, title="Code Security Scan")
        
        logger.info(f"Security scan found {len(vulnerabilities)} vulnerabilities")
        return result

    def code_semantic_correctness(self, prompt, code_response, reference_code, 
                                  save_json=False, return_type="dict"):
        """
        Evaluate semantic similarity between generated and reference code.
        """
        if not self.embedder:
            raise ValueError("Embedder model is required for semantic correctness evaluation")
        
        code_emb = self.embedder.encode([code_response], convert_to_numpy=True)
        ref_emb = self.embedder.encode([reference_code], convert_to_numpy=True)
        
        similarity = float(util.cos_sim(code_emb, ref_emb).numpy()[0][0])
        
        result = {
            "prompt": prompt,
            "generated_code": code_response,
            "reference_code": reference_code,
            "semantic_similarity": similarity,
            "semantically_correct": similarity >= 0.65
        }
        
        if save_json and self.save_json:
            self.save_json(result, test_name="code_semantic_correctness")
        if return_type in ["table", "both"] and self.display_table:
            self.display_table(result, title="Code Semantic Correctness")
        
        logger.info(f"Code semantic similarity: {similarity:.3f}")
        return result

    def comprehensive_code_evaluation(self, prompt, code_response, reference_code=None,
                                     test_cases=None, language="python", 
                                     save_json=False, return_type="dict"):
        """
        Run all code evaluation metrics in one comprehensive test.

        Implements the CCE formula from the paper:
        CCE = α*SV + β*EPR + γ*CQS + δ*(1-SRS) + ϵ*SCC

        Where:
        - SV: Syntax Validity (0 or 1)
        - EPR: Execution Pass Rate [0, 1]
        - CQS: Code Quality Score [0, 1]
        - SRS: Security Risk Score [0, 1] - inverted as (1-SRS)
        - SCC: Semantic Correctness [0, 1]

        Default weights: α=0.15, β=0.35, γ=0.20, δ=0.15, ϵ=0.15
        """
        results = {
            "prompt": prompt,
            "generated_code": code_response,
            "language": language
        }
        
        # Syntax validation
        syntax_result = self.code_syntax_validity(code_response, language, return_type="dict")
        results["syntax_valid"] = syntax_result["syntax_valid"]
        results["syntax_error"] = syntax_result.get("error")
        
        # Code quality
        quality_result = self.code_quality_metrics(code_response, language, return_type="dict")
        results["quality_score"] = quality_result["quality_score"]
        results["quality_metrics"] = quality_result["metrics"]
        
        # Security scan
        security_result = self.code_security_scan(code_response, language, return_type="dict")
        results["is_secure"] = security_result["is_secure"]
        results["vulnerabilities"] = security_result["vulnerabilities"]
        results["srs"] = security_result["srs"]

        # Execution tests (if provided)
        if test_cases:
            exec_result = self.code_execution_test(code_response, test_cases, language, return_type="dict")
            results["pass_rate"] = exec_result["pass_rate"]
            results["test_results"] = exec_result["test_results"]

        # Semantic correctness (if reference provided)
        if reference_code and self.embedder:
            semantic_result = self.code_semantic_correctness(prompt, code_response, reference_code, return_type="dict")
            results["semantic_similarity"] = semantic_result["semantic_similarity"]

        # Calculate CCE using the paper's formula
        # CCE = α*SV + β*EPR + γ*CQS + δ*(1-SRS) + ϵ*SCC
        # Default weights
        alpha = 0.15  # Syntax validity weight
        beta = 0.35   # Execution pass rate weight
        gamma = 0.20  # Code quality weight
        delta = 0.15  # Security weight
        epsilon = 0.15  # Semantic correctness weight

        # Extract normalized metrics
        sv = 1.0 if results.get("syntax_valid") else 0.0
        epr = results.get("pass_rate", 1.0)  # Default to 1.0 if no test cases
        cqs = results.get("quality_score", 0) / 100.0  # Normalize to [0, 1]
        srs = results.get("srs", 0.0)  # Security risk score
        scc = results.get("semantic_similarity", 1.0)  # Default to 1.0 if no reference

        # Calculate CCE (note: SRS is inverted)
        cce = (alpha * sv +
               beta * epr +
               gamma * cqs +
               delta * (1 - srs) +
               epsilon * scc)

        # Store both normalized (0-1) and percentage (0-100) scores
        results["cce_score"] = cce  # Normalized score [0, 1]
        results["overall_score"] = cce * 100  # Percentage score [0, 100]

        # Store individual normalized components for transparency
        results["metrics_breakdown"] = {
            "sv": sv,
            "epr": epr,
            "cqs": cqs,
            "srs": srs,
            "security_score": 1 - srs,  # Inverted for interpretation
            "scc": scc
        }

        if save_json and self.save_json:
            self.save_json(results, test_name="comprehensive_code_evaluation")
        if return_type in ["table", "both"] and self.display_table:
            self.display_table(results, title="Comprehensive Code Evaluation")
        
        logger.info(f"CCE Score: {cce:.4f} (Overall: {cce * 100:.2f}/100)")
        return results
