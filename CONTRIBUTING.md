# Contributing to Marketing Attribution Models

We welcome contributions to the Marketing Attribution Models framework! This project helps organizations understand true marketing ROI through advanced attribution modeling.

## Getting Started

1. **Fork the repository** and clone your fork locally
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Set up environment variables** in `.env` file
4. **Run tests** to ensure everything works: `pytest`

## Development Guidelines

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all classes and functions
- Maintain test coverage above 90%

### Attribution Model Development
- All attribution models must inherit from `BaseAttributionModel`
- Include statistical significance testing for model outputs
- Provide confidence intervals for attribution results
- Document model assumptions and limitations

### Data Privacy & Compliance
- Ensure GDPR compliance in all data processing
- Use anonymized customer identifiers only
- Implement proper data retention policies
- Document privacy implications of new features

## Contribution Process

### 1. Issue Creation
- Search existing issues before creating new ones
- Use issue templates for bug reports and feature requests
- Provide detailed context for attribution modeling use cases

### 2. Pull Request Process
- Create feature branches from `main`
- Write descriptive commit messages
- Include tests for all new functionality
- Update documentation as needed

### 3. Code Review
- All changes require peer review
- Statistical models need validation from data science team
- Performance impacts must be documented
- Security review required for data access changes

## Testing Requirements

### Unit Tests
- Test individual attribution algorithms
- Mock external API dependencies
- Validate statistical calculations
- Test edge cases and error handling

### Integration Tests
- Test complete attribution pipelines
- Validate API integrations
- Test dashboard functionality
- Verify data export formats

### Performance Tests
- Benchmark attribution model execution time
- Test scalability with large datasets
- Monitor memory usage patterns
- Validate real-time processing capabilities

## Documentation

### Code Documentation
- Document all parameters and return values
- Explain attribution model mathematics
- Provide usage examples
- Include performance characteristics

### User Documentation
- Update README for new features
- Create tutorial notebooks for complex workflows
- Document API endpoints and parameters
- Maintain changelog for releases

## Marketing Attribution Specific Guidelines

### Model Implementation
- Include removal effect calculations for Markov models
- Implement proper normalization for Shapley values
- Validate against known attribution benchmarks
- Provide model comparison utilities

### Data Requirements
- Document required data schema
- Specify minimum data quality thresholds
- Handle missing touchpoint data gracefully
- Implement data validation pipelines

### Business Impact Validation
- Test models against A/B test results when available
- Compare with incrementality measurements
- Validate ROI predictions with actual outcomes
- Document model accuracy and confidence levels

## Support

- **Questions**: Open a discussion in GitHub Discussions
- **Bugs**: Create an issue with reproduction steps
- **Feature Requests**: Use the feature request template
- **Security Issues**: Email security@company.com privately

Thank you for contributing to better marketing attribution! 🚀