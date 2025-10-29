"""
Input Sanitizer - UIP Step 0 Component
======================================

Input validation and sanitization for UIP pipeline entry.
Integrates LOGOS Validator Hub architecture with V2 framework.

Adapted from: V2_Possible_Gap_Fillers/ontological validator/logos_validator_hub.py
"""

from protocols.shared.system_imports import *
from protocols.shared.message_formats import UIPRequest
from typing import Dict, List, Any, Optional
import re
import html


class BaseValidator:
    """Base class for all input validators"""
    
    def name(self) -> str:
        return self.__class__.__name__
    
    def validate(self, content: str) -> Dict[str, Any]:
        """
        Validate content and return result with details
        
        Returns:
            Dict with 'valid': bool, 'issues': List[str], 'metadata': Dict
        """
        raise NotImplementedError("Each validator must implement the validate method")


class SecurityValidator(BaseValidator):
    """Security-focused input validation"""
    
    def __init__(self):
        self.dangerous_patterns = [
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'<script[^>]*>',
            r'javascript:',
            r'data:text/html',
            r'\.\.\/\.\.\/\.\.\/',  # Path traversal
            r'DROP\s+TABLE',       # SQL injection
            r'UNION\s+SELECT',
        ]
    
    def validate(self, content: str) -> Dict[str, Any]:
        issues = []
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Dangerous pattern detected: {pattern}")
        
        # Length check
        if len(content) > 50000:  # 50KB limit
            issues.append(f"Content too long: {len(content)} characters")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'metadata': {'length': len(content)}
        }


class SyntaxValidator(BaseValidator):
    """Basic syntax and encoding validation"""
    
    def validate(self, content: str) -> Dict[str, Any]:
        issues = []
        
        try:
            # Test UTF-8 encoding
            content.encode('utf-8')
        except UnicodeError as e:
            issues.append(f"Invalid UTF-8 encoding: {e}")
        
        # Basic parentheses balance check
        paren_balance = 0
        for char in content:
            if char == '(':
                paren_balance += 1
            elif char == ')':
                paren_balance -= 1
                if paren_balance < 0:
                    issues.append("Unbalanced parentheses detected")
                    break
        
        if paren_balance != 0:
            issues.append("Unbalanced parentheses detected")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'metadata': {'paren_balance': paren_balance}
        }


class ContentValidator(BaseValidator):
    """Content appropriateness validation"""
    
    def __init__(self):
        self.blocked_terms = [
            'malicious_code',
            'system_exploit',
            'bypass_security',
        ]
    
    def validate(self, content: str) -> Dict[str, Any]:
        issues = []
        
        content_lower = content.lower()
        
        # Check for blocked terms
        for term in self.blocked_terms:
            if term in content_lower:
                issues.append(f"Blocked term detected: {term}")
        
        # Empty content check
        if not content.strip():
            issues.append("Empty or whitespace-only content")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'metadata': {'blocked_terms_found': len([t for t in self.blocked_terms if t in content_lower])}
        }


class InputSanitizer:
    """Main input sanitization engine adapted from LOGOS Validator Hub"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validators: List[BaseValidator] = []
        self.emergency_halt = False
        
        # Initialize default validators
        self._initialize_validators()
    
    def _initialize_validators(self):
        """Initialize default validator set"""
        self.validators = [
            SecurityValidator(),
            SyntaxValidator(), 
            ContentValidator(),
        ]
        
        self.logger.info(f"Input sanitizer initialized with {len(self.validators)} validators")
    
    def register_validator(self, validator: BaseValidator):
        """Register additional validator"""
        self.validators.append(validator)
        self.logger.info(f"Registered validator: {validator.name()}")
    
    def sanitize_input(self, request: UIPRequest) -> Dict[str, Any]:
        """
        Sanitize and validate input request
        
        Args:
            request: UIP request to validate
            
        Returns:
            Dict with validation results and sanitized content
        """
        if self.emergency_halt:
            return {
                'valid': False,
                'sanitized_content': '',
                'issues': ['System in emergency halt state'],
                'metadata': {'emergency_halt': True}
            }
        
        content = request.user_input
        all_issues = []
        all_metadata = {}
        
        # Run through all validators
        for validator in self.validators:
            try:
                result = validator.validate(content)
                
                if not result['valid']:
                    all_issues.extend(result['issues'])
                
                # Merge metadata
                validator_name = validator.name()
                all_metadata[validator_name] = result['metadata']
                
            except Exception as e:
                self.logger.error(f"Validator {validator.name()} failed: {e}")
                all_issues.append(f"Validator error: {validator.name()}")
        
        # Basic sanitization if validation passes
        sanitized_content = content
        if len(all_issues) == 0:
            # HTML escape for safety
            sanitized_content = html.escape(content)
            
            # Normalize whitespace
            sanitized_content = re.sub(r'\s+', ' ', sanitized_content.strip())
        
        result = {
            'valid': len(all_issues) == 0,
            'sanitized_content': sanitized_content,
            'issues': all_issues,
            'metadata': {
                'original_length': len(content),
                'sanitized_length': len(sanitized_content),
                'validators_run': len(self.validators),
                'validator_details': all_metadata
            }
        }
        
        # Log validation result
        if result['valid']:
            self.logger.debug(f"Input validation passed for {len(content)} characters")
        else:
            self.logger.warning(f"Input validation failed: {len(all_issues)} issues found")
        
        return result
    
    def get_validator_summary(self) -> List[str]:
        """Get summary of registered validators"""
        return [v.name() for v in self.validators]
    
    def set_emergency_halt(self, halt: bool = True):
        """Set emergency halt state"""
        self.emergency_halt = halt
        self.logger.warning(f"Emergency halt {'activated' if halt else 'deactivated'}")


# Global sanitizer instance
input_sanitizer = InputSanitizer()


__all__ = [
    'BaseValidator',
    'SecurityValidator', 
    'SyntaxValidator',
    'ContentValidator',
    'InputSanitizer',
    'input_sanitizer'
]