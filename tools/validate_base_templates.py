#!/usr/bin/env python3
# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Validation script for base templates.

This script validates that base templates follow required structure,
have proper imports, correct function signatures, and complete docstrings.
"""

import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List


class TemplateValidator:
    """Validator for base template files."""

    def __init__(self, template_dir: Path):
        self.template_dir = template_dir
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {
            "templates_checked": 0,
            "errors": 0,
            "warnings": 0,
            "checks_passed": 0,
        }

    def add_error(self, template: str, check: str, message: str):
        """Add an error to the results."""
        self.errors.append({"template": template, "check": check, "message": message})
        self.results["errors"] += 1

    def add_warning(self, template: str, check: str, message: str):
        """Add a warning to the results."""
        self.warnings.append({"template": template, "check": check, "message": message})
        self.results["warnings"] += 1

    def add_pass(self):
        """Increment passed checks counter."""
        self.results["checks_passed"] += 1

    def validate_structure(self) -> bool:
        """Validate that required template files exist with proper structure."""
        print("[INFO] Validating template structure...")
        
        required_templates = [
            "application_function_template.py",
            "circuit_function_template.py",
        ]
        
        all_valid = True
        for template_name in required_templates:
            template_path = self.template_dir / template_name
            
            if not template_path.exists():
                self.add_error(
                    template_name,
                    "structure",
                    f"Template file not found: {template_path}"
                )
                all_valid = False
                continue
            
            # Check file is not empty
            if template_path.stat().st_size == 0:
                self.add_error(
                    template_name,
                    "structure",
                    "Template file is empty"
                )
                all_valid = False
                continue
            
            # Parse the file to check it's valid Python
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
                    self.add_pass()
                    print(f"  [PASS] {template_name} structure is valid")
            except SyntaxError as e:
                self.add_error(
                    template_name,
                    "structure",
                    f"Syntax error in template: {e}"
                )
                all_valid = False
        
        self.results["templates_checked"] = len(required_templates)
        return all_valid

    def validate_imports(self) -> bool:
        """Validate that templates can be imported without errors."""
        print("\n[INFO] Validating template imports...")
        
        templates = [
            "application_function_template.py",
            "circuit_function_template.py",
        ]
        
        all_valid = True
        for template_name in templates:
            template_path = self.template_dir / template_name
            
            if not template_path.exists():
                continue
            
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    template_name.replace('.py', ''),
                    template_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.add_pass()
                    print(f"  [PASS] {template_name} imports successfully")
                else:
                    self.add_error(
                        template_name,
                        "imports",
                        "Could not create module spec"
                    )
                    all_valid = False
            except Exception as e:
                self.add_error(
                    template_name,
                    "imports",
                    f"Import error: {type(e).__name__}: {e}"
                )
                all_valid = False
        
        return all_valid

    def validate_signatures(self) -> bool:
        """Validate that required functions/classes exist with correct signatures."""
        print("\n[INFO] Validating function/class signatures...")
        
        # Expected signatures for each template
        expected_signatures = {
            "application_function_template.py": {
                "functions": ["run_function"],
                "classes": [],
            },
            "circuit_function_template.py": {
                "functions": [],
                "classes": ["CircuitFunction"],
            },
        }
        
        all_valid = True
        for template_name, expectations in expected_signatures.items():
            template_path = self.template_dir / template_name
            
            if not template_path.exists():
                continue
            
            try:
                # Parse the AST
                with open(template_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                # Find all function and class definitions
                functions = [node.name for node in ast.walk(tree) 
                           if isinstance(node, ast.FunctionDef)]
                classes = [node.name for node in ast.walk(tree) 
                         if isinstance(node, ast.ClassDef)]
                
                # Check expected functions exist
                for expected_func in expectations["functions"]:
                    if expected_func in functions:
                        self.add_pass()
                        print(f"  [PASS] {template_name}: function '{expected_func}' found")
                    else:
                        self.add_error(
                            template_name,
                            "signatures",
                            f"Required function '{expected_func}' not found"
                        )
                        all_valid = False
                
                # Check expected classes exist
                for expected_class in expectations["classes"]:
                    if expected_class in classes:
                        self.add_pass()
                        print(f"  [PASS] {template_name}: class '{expected_class}' found")
                    else:
                        self.add_error(
                            template_name,
                            "signatures",
                            f"Required class '{expected_class}' not found"
                        )
                        all_valid = False
                        
            except Exception as e:
                self.add_error(
                    template_name,
                    "signatures",
                    f"Error parsing template: {e}"
                )
                all_valid = False
        
        return all_valid

    def validate_docstrings(self) -> bool:
        """Validate that functions and classes have proper docstrings."""
        print("\n[INFO] Validating docstrings...")
        
        templates = [
            "application_function_template.py",
            "circuit_function_template.py",
        ]
        
        all_valid = True
        for template_name in templates:
            template_path = self.template_dir / template_name
            
            if not template_path.exists():
                continue
            
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                # Check module docstring
                module_docstring = ast.get_docstring(tree)
                if not module_docstring:
                    self.add_warning(
                        template_name,
                        "docstrings",
                        "Module docstring is missing"
                    )
                else:
                    self.add_pass()
                    print(f"  [PASS] {template_name}: module docstring present")
                
                # Check function and class docstrings
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        # Skip private functions/methods
                        if node.name.startswith('_') and not node.name.startswith('__'):
                            continue
                        
                        docstring = ast.get_docstring(node)
                        if not docstring:
                            self.add_warning(
                                template_name,
                                "docstrings",
                                f"{type(node).__name__} '{node.name}' is missing a docstring"
                            )
                        else:
                            # Check for basic docstring components
                            has_args = "Args:" in docstring or "Parameters:" in docstring
                            has_returns = "Returns:" in docstring
                            
                            if isinstance(node, ast.FunctionDef):
                                # Check if function has parameters
                                has_params = len(node.args.args) > 0
                                if has_params and not has_args:
                                    self.add_warning(
                                        template_name,
                                        "docstrings",
                                        f"Function '{node.name}' has parameters but no Args section"
                                    )
                                
                                # Check if function has return annotation
                                if node.returns and not has_returns:
                                    self.add_warning(
                                        template_name,
                                        "docstrings",
                                        f"Function '{node.name}' has return type but no Returns section"
                                    )
                            
                            self.add_pass()
                            print(f"  [PASS] {template_name}: '{node.name}' has docstring")
                            
            except Exception as e:
                self.add_error(
                    template_name,
                    "docstrings",
                    f"Error checking docstrings: {e}"
                )
                all_valid = False
        
        return all_valid

    def print_summary(self):
        """Print a summary of validation results."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Templates checked: {self.results['templates_checked']}")
        print(f"Checks passed: {self.results['checks_passed']}")
        print(f"Errors: {self.results['errors']}")
        print(f"Warnings: {self.results['warnings']}")
        
        if self.errors:
            print("\n[ERROR] ERRORS:")
            for error in self.errors:
                print(f"  [{error['template']}] {error['check']}: {error['message']}")
        
        if self.warnings:
            print("\n[WARN] WARNINGS:")
            for warning in self.warnings:
                print(f"  [{warning['template']}] {warning['check']}: {warning['message']}")
        
        if not self.errors and not self.warnings:
            print("\n[PASS] All validations passed!")
        elif not self.errors:
            print("\n[PASS] All critical validations passed (warnings present)")
        
        print("="*60)


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate base template files"
    )
    parser.add_argument(
        "--check-structure",
        action="store_true",
        help="Check template file structure"
    )
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help="Check that templates can be imported"
    )
    parser.add_argument(
        "--check-signatures",
        action="store_true",
        help="Check function/class signatures"
    )
    parser.add_argument(
        "--check-docstrings",
        action="store_true",
        help="Check docstring completeness"
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("base_templates"),
        help="Path to base templates directory"
    )
    
    args = parser.parse_args()
    
    # If no specific checks requested, run all
    run_all = not any([
        args.check_structure,
        args.check_imports,
        args.check_signatures,
        args.check_docstrings,
    ])
    
    validator = TemplateValidator(args.template_dir)
    
    all_passed = True
    
    if run_all or args.check_structure:
        if not validator.validate_structure():
            all_passed = False
    
    if run_all or args.check_imports:
        if not validator.validate_imports():
            all_passed = False
    
    if run_all or args.check_signatures:
        if not validator.validate_signatures():
            all_passed = False
    
    if run_all or args.check_docstrings:
        if not validator.validate_docstrings():
            all_passed = False
    
    validator.print_summary()
    
    # Exit with error code if validation failed
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
