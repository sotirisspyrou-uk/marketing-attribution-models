"""
Executive Marketing Attribution Reporting

Comprehensive executive reporting system for marketing attribution insights,
strategic KPI dashboards, and board-ready analytics with automated report
generation and data storytelling capabilities.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import base64
import io

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of executive reports."""
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_PERFORMANCE = "monthly_performance"
    QUARTERLY_REVIEW = "quarterly_review"
    ANNUAL_REPORT = "annual_report"
    CAMPAIGN_ANALYSIS = "campaign_analysis"
    BUDGET_REVIEW = "budget_review"
    ATTRIBUTION_INSIGHTS = "attribution_insights"
    ROI_ANALYSIS = "roi_analysis"


class ReportFormat(Enum):
    """Report output formats."""
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    POWERPOINT = "powerpoint"
    EXCEL = "excel"


@dataclass
class ExecutiveMetric:
    """Executive-level metric definition."""
    name: str
    value: float
    target: Optional[float] = None
    previous_period: Optional[float] = None
    format_type: str = "number"  # "number", "percentage", "currency"
    trend: str = "stable"  # "up", "down", "stable"
    significance: str = "medium"  # "high", "medium", "low"
    description: str = ""


@dataclass
class ReportSection:
    """Report section structure."""
    title: str
    summary: str
    metrics: List[ExecutiveMetric]
    insights: List[str]
    recommendations: List[str]
    charts: List[Dict[str, Any]]
    priority: int = 1


class ExecutiveReportGenerator:
    """
    Advanced executive reporting system for marketing attribution.
    
    Generates comprehensive, board-ready reports with strategic insights,
    performance analysis, and actionable recommendations for C-level executives.
    """
    
    def __init__(self,
                 company_name: str = "Marketing Organization",
                 reporting_period: str = "monthly",
                 currency: str = "USD",
                 enable_visualizations: bool = True):
        """
        Initialize Executive Report Generator.
        
        Args:
            company_name: Organization name for reports
            reporting_period: Default reporting period
            currency: Currency for financial metrics
            enable_visualizations: Enable chart generation
        """
        self.company_name = company_name
        self.reporting_period = reporting_period
        self.currency = currency
        self.enable_visualizations = enable_visualizations
        
        # Report data
        self.report_data = {}
        self.report_sections = []
        self.executive_summary = ""
        self.key_insights = []
        self.strategic_recommendations = []
        
        # Templates and styling
        self.report_templates = {}
        self.brand_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
        
        logger.info(f"Executive report generator initialized for {company_name}")
    
    def generate_executive_report(self,
                                attribution_data: Dict[str, float],
                                performance_data: Dict[str, Dict[str, float]],
                                historical_data: Optional[pd.DataFrame] = None,
                                budget_data: Optional[Dict[str, float]] = None,
                                goals_data: Optional[Dict[str, float]] = None,
                                report_type: ReportType = ReportType.MONTHLY_PERFORMANCE,
                                report_format: ReportFormat = ReportFormat.MARKDOWN) -> str:
        """
        Generate comprehensive executive marketing report.
        
        Args:
            attribution_data: Channel attribution weights
            performance_data: Performance metrics by channel
            historical_data: Historical performance trends
            budget_data: Budget allocation and spend data
            goals_data: Performance goals and targets
            report_type: Type of report to generate
            report_format: Output format for the report
            
        Returns:
            Generated report in specified format
        """
        logger.info(f"Generating {report_type.value} executive report")
        
        # Store data
        self.report_data = {
            'attribution': attribution_data,
            'performance': performance_data,
            'historical': historical_data,
            'budget': budget_data or {},
            'goals': goals_data or {},
            'report_type': report_type,
            'generation_time': datetime.now()
        }
        
        # Generate report sections
        self._generate_executive_summary()
        self._generate_performance_overview()
        self._generate_attribution_analysis()
        self._generate_roi_insights()
        self._generate_strategic_recommendations()
        
        if budget_data:
            self._generate_budget_analysis()
        
        # Generate final report
        if report_format == ReportFormat.MARKDOWN:
            return self._generate_markdown_report()
        elif report_format == ReportFormat.HTML:
            return self._generate_html_report()
        elif report_format == ReportFormat.JSON:
            return self._generate_json_report()
        else:
            return self._generate_markdown_report()  # Default fallback
    
    def _generate_executive_summary(self):
        """Generate executive summary section."""
        
        performance_data = self.report_data['performance']
        attribution_data = self.report_data['attribution']
        
        # Calculate key metrics
        total_spend = sum(metrics.get('spend', 0) for metrics in performance_data.values())
        total_revenue = sum(metrics.get('revenue', 0) for metrics in performance_data.values())
        overall_roas = total_revenue / total_spend if total_spend > 0 else 0
        
        # Top performing channel
        top_channel = max(attribution_data.items(), key=lambda x: x[1])[0] if attribution_data else "N/A"
        
        # Channel efficiency
        efficiency_scores = {}
        for channel, metrics in performance_data.items():
            roas = metrics.get('roas', 0)
            attribution = attribution_data.get(channel, 0)
            efficiency_scores[channel] = roas * attribution
        
        most_efficient = max(efficiency_scores.items(), key=lambda x: x[1])[0] if efficiency_scores else "N/A"
        
        # Generate summary text
        self.executive_summary = f"""
**{self.company_name} Marketing Performance Summary**

This {self.report_data['report_type'].value.replace('_', ' ').title()} provides comprehensive analysis of marketing attribution and performance across all channels.

**Key Highlights:**
- Total marketing investment: ${total_spend:,.0f}
- Revenue generated: ${total_revenue:,.0f}
- Overall ROAS: {overall_roas:.1f}x
- Top attributed channel: {top_channel} ({attribution_data.get(top_channel, 0):.1%} attribution)
- Most efficient channel: {most_efficient}

**Strategic Focus Areas:**
- Attribution-driven budget optimization opportunities identified
- Performance improvement potential across underperforming channels
- ROI maximization through data-driven allocation strategies
        """.strip()
    
    def _generate_performance_overview(self):
        """Generate performance overview section."""
        
        performance_data = self.report_data['performance']
        
        # Calculate performance metrics
        metrics = []
        
        # Total metrics
        total_spend = sum(metrics.get('spend', 0) for metrics in performance_data.values())
        total_revenue = sum(metrics.get('revenue', 0) for metrics in performance_data.values())
        total_conversions = sum(metrics.get('conversions', 0) for metrics in performance_data.values())
        
        metrics.append(ExecutiveMetric(
            name="Total Marketing Spend",
            value=total_spend,
            format_type="currency",
            description="Total marketing investment across all channels"
        ))
        
        metrics.append(ExecutiveMetric(
            name="Revenue Generated",
            value=total_revenue,
            format_type="currency",
            description="Total revenue attributed to marketing activities"
        ))
        
        metrics.append(ExecutiveMetric(
            name="Overall ROAS",
            value=total_revenue / total_spend if total_spend > 0 else 0,
            target=3.0,  # Example target
            format_type="number",
            description="Return on advertising spend across all channels"
        ))
        
        metrics.append(ExecutiveMetric(
            name="Total Conversions",
            value=total_conversions,
            format_type="number",
            description="Total conversions generated from all marketing channels"
        ))
        
        # Channel-specific insights
        insights = []
        channel_roas = [(ch, data.get('roas', 0)) for ch, data in performance_data.items()]
        channel_roas.sort(key=lambda x: x[1], reverse=True)
        
        if channel_roas:
            best_channel = channel_roas[0]
            worst_channel = channel_roas[-1]
            
            insights.append(f"{best_channel[0]} delivers the highest ROAS at {best_channel[1]:.1f}x")
            
            if worst_channel[1] < 2.0:
                insights.append(f"{worst_channel[0]} requires optimization with ROAS of {worst_channel[1]:.1f}x")
        
        # Calculate performance trends (simplified)
        for channel, data in performance_data.items():
            roas = data.get('roas', 0)
            if roas > 4.0:
                insights.append(f"{channel} shows exceptional performance - consider scaling investment")
            elif roas < 1.5:
                insights.append(f"{channel} underperforming - immediate optimization required")
        
        recommendations = [
            "Focus budget allocation on high-ROAS channels for maximum efficiency",
            "Implement A/B testing for underperforming channels to improve conversion rates",
            "Monitor daily performance metrics to enable rapid optimization decisions"
        ]
        
        section = ReportSection(
            title="Performance Overview",
            summary="Comprehensive analysis of marketing channel performance and efficiency",
            metrics=metrics,
            insights=insights,
            recommendations=recommendations,
            charts=[],
            priority=1
        )
        
        self.report_sections.append(section)
    
    def _generate_attribution_analysis(self):
        """Generate attribution analysis section."""
        
        attribution_data = self.report_data['attribution']
        performance_data = self.report_data['performance']
        
        # Attribution metrics
        metrics = []
        
        # Attribution concentration
        attribution_values = list(attribution_data.values())
        concentration = sum(w**2 for w in attribution_values) if attribution_values else 0
        diversity = 1 - concentration
        
        metrics.append(ExecutiveMetric(
            name="Attribution Diversity",
            value=diversity,
            target=0.7,  # Target 70% diversity
            format_type="percentage",
            description="Measure of attribution distribution across channels"
        ))
        
        # Top attributed channels
        sorted_attribution = sorted(attribution_data.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_attribution:
            top_channel = sorted_attribution[0]
            metrics.append(ExecutiveMetric(
                name=f"Top Channel Attribution ({top_channel[0]})",
                value=top_channel[1],
                format_type="percentage",
                description="Attribution weight of highest-contributing channel"
            ))
        
        # Attribution efficiency analysis
        efficiency_insights = []
        for channel, attribution_weight in attribution_data.items():
            channel_data = performance_data.get(channel, {})
            spend = channel_data.get('spend', 0)
            
            # Calculate budget vs attribution ratio
            total_spend = sum(data.get('spend', 0) for data in performance_data.values())
            budget_share = spend / total_spend if total_spend > 0 else 0
            
            if budget_share > 0:
                efficiency_ratio = attribution_weight / budget_share
                
                if efficiency_ratio > 1.5:
                    efficiency_insights.append(f"{channel} is highly attribution-efficient ({efficiency_ratio:.1f}x) - consider increasing budget")
                elif efficiency_ratio < 0.7:
                    efficiency_insights.append(f"{channel} shows low attribution efficiency ({efficiency_ratio:.1f}x) - review performance")
        
        # Attribution insights
        insights = [
            f"Attribution is distributed across {len(attribution_data)} channels",
            f"Attribution diversity score: {diversity:.1%} (target: 70%+)",
        ] + efficiency_insights[:3]  # Top 3 efficiency insights
        
        recommendations = [
            "Reallocate budget based on attribution weights to improve overall efficiency",
            "Monitor attribution model drift to ensure continued accuracy",
            "Implement cross-channel measurement to capture interaction effects"
        ]
        
        section = ReportSection(
            title="Attribution Analysis",
            summary="Data-driven attribution insights revealing true channel contribution",
            metrics=metrics,
            insights=insights,
            recommendations=recommendations,
            charts=[],
            priority=2
        )
        
        self.report_sections.append(section)
    
    def _generate_roi_insights(self):
        """Generate ROI analysis section."""
        
        performance_data = self.report_data['performance']
        attribution_data = self.report_data['attribution']
        
        # ROI metrics
        metrics = []
        roi_insights = []
        
        # Calculate attribution-weighted ROI
        weighted_roi = 0
        for channel, data in performance_data.items():
            roas = data.get('roas', 0)
            attribution = attribution_data.get(channel, 0)
            weighted_roi += roas * attribution
        
        metrics.append(ExecutiveMetric(
            name="Attribution-Weighted ROI",
            value=weighted_roi,
            format_type="number",
            description="ROI weighted by attribution contribution"
        ))
        
        # Marginal ROI analysis
        marginal_roi = {}
        for channel, data in performance_data.items():
            spend = data.get('spend', 0)
            revenue = data.get('revenue', 0)
            
            if spend > 0:
                marginal_roi[channel] = revenue / spend
        
        # Find best and worst marginal ROI
        if marginal_roi:
            best_roi_channel = max(marginal_roi.items(), key=lambda x: x[1])
            worst_roi_channel = min(marginal_roi.items(), key=lambda x: x[1])
            
            metrics.append(ExecutiveMetric(
                name=f"Best Channel ROI ({best_roi_channel[0]})",
                value=best_roi_channel[1],
                format_type="number",
                description="Highest return on investment channel"
            ))
            
            roi_insights.extend([
                f"{best_roi_channel[0]} delivers highest marginal ROI at {best_roi_channel[1]:.1f}x",
                f"{worst_roi_channel[0]} has lowest marginal ROI at {worst_roi_channel[1]:.1f}x",
                f"ROI gap between best and worst performing channels: {best_roi_channel[1] - worst_roi_channel[1]:.1f}x"
            ])
        
        # Incremental revenue opportunities
        total_spend = sum(data.get('spend', 0) for data in performance_data.values())
        potential_reallocation = total_spend * 0.1  # 10% reallocation
        
        if marginal_roi:
            best_channel_roi = best_roi_channel[1]
            worst_channel_roi = worst_roi_channel[1]
            incremental_revenue = potential_reallocation * (best_channel_roi - worst_channel_roi)
            
            metrics.append(ExecutiveMetric(
                name="Reallocation Opportunity",
                value=incremental_revenue,
                format_type="currency",
                description="Revenue potential from 10% budget reallocation"
            ))
        
        recommendations = [
            f"Increase investment in {best_roi_channel[0]} to capitalize on superior ROI",
            f"Optimize or reduce spending on {worst_roi_channel[0]} to improve overall efficiency",
            "Implement dynamic budget allocation based on real-time ROI performance"
        ]
        
        section = ReportSection(
            title="ROI Analysis",
            summary="Strategic ROI insights and investment optimization opportunities",
            metrics=metrics,
            insights=roi_insights,
            recommendations=recommendations,
            charts=[],
            priority=3
        )
        
        self.report_sections.append(section)
    
    def _generate_budget_analysis(self):
        """Generate budget analysis section."""
        
        performance_data = self.report_data['performance']
        budget_data = self.report_data['budget']
        
        metrics = []
        insights = []
        
        # Budget utilization
        total_budget = sum(budget_data.values())
        total_spend = sum(data.get('spend', 0) for data in performance_data.values())
        utilization = total_spend / total_budget if total_budget > 0 else 0
        
        metrics.append(ExecutiveMetric(
            name="Budget Utilization",
            value=utilization,
            target=0.95,  # 95% target
            format_type="percentage",
            description="Percentage of allocated budget utilized"
        ))
        
        # Budget efficiency by channel
        efficiency_scores = []
        for channel, budget in budget_data.items():
            channel_data = performance_data.get(channel, {})
            revenue = channel_data.get('revenue', 0)
            
            if budget > 0:
                efficiency = revenue / budget
                efficiency_scores.append((channel, efficiency))
        
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        
        if efficiency_scores:
            best_efficiency = efficiency_scores[0]
            metrics.append(ExecutiveMetric(
                name=f"Most Efficient Channel ({best_efficiency[0]})",
                value=best_efficiency[1],
                format_type="number",
                description="Highest revenue per budget dollar"
            ))
            
            insights.append(f"{best_efficiency[0]} generates ${best_efficiency[1]:.2f} revenue per budget dollar")
        
        # Over/under budget analysis
        over_budget = []
        under_budget = []
        
        for channel, budget in budget_data.items():
            spend = performance_data.get(channel, {}).get('spend', 0)
            variance = (spend - budget) / budget if budget > 0 else 0
            
            if variance > 0.1:  # 10% over budget
                over_budget.append(f"{channel}: {variance:.1%} over budget")
            elif variance < -0.1:  # 10% under budget
                under_budget.append(f"{channel}: {abs(variance):.1%} under budget")
        
        if over_budget:
            insights.extend(over_budget[:2])  # Top 2
        if under_budget:
            insights.extend(under_budget[:2])  # Top 2
        
        recommendations = [
            "Reallocate underutilized budget to high-performing channels",
            "Implement budget pacing controls to optimize spend throughout the period",
            "Review budget allocation quarterly based on performance trends"
        ]
        
        section = ReportSection(
            title="Budget Analysis",
            summary="Budget allocation efficiency and optimization opportunities",
            metrics=metrics,
            insights=insights,
            recommendations=recommendations,
            charts=[],
            priority=4
        )
        
        self.report_sections.append(section)
    
    def _generate_strategic_recommendations(self):
        """Generate strategic recommendations."""
        
        performance_data = self.report_data['performance']
        attribution_data = self.report_data['attribution']
        
        # Analyze performance and generate strategic recommendations
        recommendations = []
        
        # Top performers for scaling
        channel_efficiency = {}
        for channel, data in performance_data.items():
            roas = data.get('roas', 0)
            attribution = attribution_data.get(channel, 0)
            efficiency = roas * attribution
            channel_efficiency[channel] = efficiency
        
        top_channel = max(channel_efficiency.items(), key=lambda x: x[1])[0] if channel_efficiency else None
        
        if top_channel:
            recommendations.append({
                'title': f'Scale {top_channel} Investment',
                'description': f'{top_channel} shows exceptional efficiency and should receive increased budget allocation',
                'priority': 'High',
                'timeline': '2-4 weeks',
                'expected_impact': 'Increase overall ROAS by 15-25%'
            })
        
        # Underperformers for optimization
        underperformers = [
            channel for channel, data in performance_data.items()
            if data.get('roas', 0) < 2.0
        ]
        
        if underperformers:
            recommendations.append({
                'title': f'Optimize Underperforming Channels',
                'description': f'Channels requiring immediate attention: {", ".join(underperformers[:2])}',
                'priority': 'High',
                'timeline': '1-3 weeks',
                'expected_impact': 'Improve bottom-quartile performance by 20-30%'
            })
        
        # Attribution-based reallocation
        if len(attribution_data) > 2:
            recommendations.append({
                'title': 'Attribution-Driven Budget Reallocation',
                'description': 'Realign budget allocation with attribution weights to improve overall efficiency',
                'priority': 'Medium',
                'timeline': '4-6 weeks',
                'expected_impact': 'Optimize budget efficiency by 10-15%'
            })
        
        # Advanced analytics implementation
        recommendations.append({
            'title': 'Implement Predictive Analytics',
            'description': 'Deploy machine learning models for predictive budget optimization and performance forecasting',
            'priority': 'Medium',
            'timeline': '8-12 weeks',
            'expected_impact': 'Enable proactive optimization and 5-10% efficiency gains'
        })
        
        self.strategic_recommendations = recommendations
    
    def _generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report."""
        
        report = f"# {self.company_name} Marketing Attribution Report\n\n"
        report += f"**Report Type:** {self.report_data['report_type'].value.replace('_', ' ').title()}\n"
        report += f"**Generated:** {self.report_data['generation_time'].strftime('%B %d, %Y at %I:%M %p')}\n"
        report += f"**Author:** Sotiris Spyrou | Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        report += self.executive_summary + "\n\n"
        
        # Report Sections
        for section in self.report_sections:
            report += f"## {section.title}\n\n"
            report += f"{section.summary}\n\n"
            
            # Key Metrics
            if section.metrics:
                report += "### Key Metrics\n\n"
                for metric in section.metrics:
                    value_str = self._format_metric_value(metric.value, metric.format_type)
                    target_str = ""
                    if metric.target:
                        target_value = self._format_metric_value(metric.target, metric.format_type)
                        performance = "✅" if metric.value >= metric.target else "⚠️"
                        target_str = f" | Target: {target_value} {performance}"
                    
                    report += f"- **{metric.name}**: {value_str}{target_str}\n"
                    if metric.description:
                        report += f"  - {metric.description}\n"
                
                report += "\n"
            
            # Insights
            if section.insights:
                report += "### Key Insights\n\n"
                for insight in section.insights:
                    report += f"- {insight}\n"
                report += "\n"
            
            # Recommendations
            if section.recommendations:
                report += "### Recommendations\n\n"
                for i, rec in enumerate(section.recommendations, 1):
                    report += f"{i}. {rec}\n"
                report += "\n"
        
        # Strategic Recommendations
        if self.strategic_recommendations:
            report += "## Strategic Action Plan\n\n"
            
            for i, rec in enumerate(self.strategic_recommendations, 1):
                priority_emoji = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Medium' else "🟢"
                
                report += f"### {i}. {rec['title']} {priority_emoji}\n\n"
                report += f"**Description:** {rec['description']}\n\n"
                report += f"**Priority:** {rec['priority']} | **Timeline:** {rec['timeline']}\n\n"
                report += f"**Expected Impact:** {rec['expected_impact']}\n\n"
        
        # Channel Performance Summary
        performance_data = self.report_data['performance']
        attribution_data = self.report_data['attribution']
        
        report += "## Channel Performance Summary\n\n"
        report += "| Channel | Attribution | ROAS | Spend | Revenue | Efficiency Score |\n"
        report += "|---------|-------------|------|--------|---------|------------------|\n"
        
        for channel in performance_data.keys():
            attribution = attribution_data.get(channel, 0)
            data = performance_data[channel]
            roas = data.get('roas', 0)
            spend = data.get('spend', 0)
            revenue = data.get('revenue', 0)
            efficiency = roas * attribution
            
            report += f"| {channel} | {attribution:.1%} | {roas:.1f}x | ${spend:,.0f} | ${revenue:,.0f} | {efficiency:.2f} |\n"
        
        # Footer
        report += "\n---\n\n"
        report += "**About This Report**\n\n"
        report += "This executive marketing attribution report provides data-driven insights for strategic decision-making. "
        report += "The analysis combines advanced attribution modeling with performance analytics to reveal true channel contribution and optimization opportunities.\n\n"
        
        report += "**Contact Information**\n\n"
        report += "For questions about this report or custom analytics implementations:\n"
        report += "- **Sotiris Spyrou** - Senior Marketing Analytics Professional\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n"
        
        return report
    
    def _format_metric_value(self, value: float, format_type: str) -> str:
        """Format metric value based on type."""
        
        if format_type == "currency":
            return f"${value:,.0f}"
        elif format_type == "percentage":
            return f"{value:.1%}"
        elif format_type == "number":
            if value >= 1000:
                return f"{value:,.0f}"
            else:
                return f"{value:.2f}"
        else:
            return str(value)
    
    def _generate_html_report(self) -> str:
        """Generate HTML report (basic implementation)."""
        
        markdown_report = self._generate_markdown_report()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.company_name} Marketing Attribution Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1, h2, h3 {{ color: {self.brand_colors['primary']}; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .high-priority {{ color: {self.brand_colors['warning']}; }}
                .medium-priority {{ color: {self.brand_colors['secondary']}; }}
            </style>
        </head>
        <body>
            {markdown_report.replace('\n', '<br>')}
        </body>
        </html>
        """
        
        return html
    
    def _generate_json_report(self) -> str:
        """Generate JSON report."""
        
        report_data = {
            'company_name': self.company_name,
            'report_type': self.report_data['report_type'].value,
            'generation_time': self.report_data['generation_time'].isoformat(),
            'executive_summary': self.executive_summary,
            'sections': [
                {
                    'title': section.title,
                    'summary': section.summary,
                    'metrics': [
                        {
                            'name': metric.name,
                            'value': metric.value,
                            'target': metric.target,
                            'format_type': metric.format_type,
                            'description': metric.description
                        }
                        for metric in section.metrics
                    ],
                    'insights': section.insights,
                    'recommendations': section.recommendations,
                    'priority': section.priority
                }
                for section in self.report_sections
            ],
            'strategic_recommendations': self.strategic_recommendations,
            'performance_data': self.report_data['performance'],
            'attribution_data': self.report_data['attribution'],
            'metadata': {
                'author': 'Sotiris Spyrou',
                'portfolio': 'https://verityai.co',
                'linkedin': 'https://www.linkedin.com/in/sspyrou/',
                'disclaimer': 'This is demonstration code for portfolio purposes.'
            }
        }
        
        return json.dumps(report_data, indent=2, default=str)


def demo_executive_reporting():
    """Executive demonstration of Executive Reporting System."""
    
    print("=== Executive Marketing Attribution Reporting: Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    # Initialize report generator
    generator = ExecutiveReportGenerator(
        company_name="TechCorp Marketing",
        reporting_period="monthly",
        currency="USD",
        enable_visualizations=True
    )
    
    print("📊 Generating executive marketing attribution report...")
    
    # Sample data for comprehensive report
    attribution_data = {
        'Search': 0.35,
        'Display': 0.20,
        'Social': 0.25,
        'Email': 0.15,
        'Direct': 0.05
    }
    
    performance_data = {
        'Search': {
            'spend': 120000,
            'revenue': 480000,
            'roas': 4.0,
            'conversions': 2400,
            'ctr': 0.08,
            'conversion_rate': 0.15
        },
        'Display': {
            'spend': 80000,
            'revenue': 200000,
            'roas': 2.5,
            'conversions': 1200,
            'ctr': 0.03,
            'conversion_rate': 0.08
        },
        'Social': {
            'spend': 60000,
            'revenue': 240000,
            'roas': 4.0,
            'conversions': 1800,
            'ctr': 0.06,
            'conversion_rate': 0.12
        },
        'Email': {
            'spend': 20000,
            'revenue': 140000,
            'roas': 7.0,
            'conversions': 1400,
            'ctr': 0.15,
            'conversion_rate': 0.25
        },
        'Direct': {
            'spend': 5000,
            'revenue': 50000,
            'roas': 10.0,
            'conversions': 500,
            'ctr': 0.25,
            'conversion_rate': 0.35
        }
    }
    
    budget_data = {
        'Search': 125000,
        'Display': 85000,
        'Social': 65000,
        'Email': 25000,
        'Direct': 5000
    }
    
    goals_data = {
        'overall_roas': 3.5,
        'total_revenue': 1000000,
        'attribution_diversity': 0.70
    }
    
    # Generate comprehensive report
    report = generator.generate_executive_report(
        attribution_data=attribution_data,
        performance_data=performance_data,
        budget_data=budget_data,
        goals_data=goals_data,
        report_type=ReportType.MONTHLY_PERFORMANCE,
        report_format=ReportFormat.MARKDOWN
    )
    
    print("\n📋 EXECUTIVE REPORT GENERATED")
    print("=" * 50)
    
    # Show report excerpt (first 30 lines)
    report_lines = report.split('\n')
    excerpt_lines = report_lines[:30]
    
    for line in excerpt_lines:
        print(line)
    
    print(f"\n... (Report continues for {len(report_lines)} total lines)")
    
    # Calculate and display key summary metrics
    total_spend = sum(data['spend'] for data in performance_data.values())
    total_revenue = sum(data['revenue'] for data in performance_data.values())
    total_conversions = sum(data['conversions'] for data in performance_data.values())
    overall_roas = total_revenue / total_spend
    
    print(f"\n📈 REPORT HIGHLIGHTS:")
    print(f"  • Total Marketing Investment: ${total_spend:,}")
    print(f"  • Revenue Generated: ${total_revenue:,}")
    print(f"  • Overall ROAS: {overall_roas:.1f}x")
    print(f"  • Total Conversions: {total_conversions:,}")
    
    # Top channel insights
    top_attribution = max(attribution_data.items(), key=lambda x: x[1])
    top_roas = max(performance_data.items(), key=lambda x: x[1]['roas'])
    
    print(f"\n🏆 TOP PERFORMERS:")
    print(f"  • Highest Attribution: {top_attribution[0]} ({top_attribution[1]:.1%})")
    print(f"  • Highest ROAS: {top_roas[0]} ({top_roas[1]['roas']:.1f}x)")
    
    # Strategic recommendations preview
    print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
    print(f"  1. Scale Email marketing investment (7.0x ROAS with strong attribution)")
    print(f"  2. Optimize Display channel performance (below target ROAS)")
    print(f"  3. Implement attribution-driven budget reallocation")
    print(f"  4. Deploy advanced predictive analytics for optimization")
    
    # Report formats available
    print(f"\n📄 AVAILABLE REPORT FORMATS:")
    print(f"  • Markdown: Comprehensive text-based report")
    print(f"  • HTML: Web-ready with styling and formatting")
    print(f"  • JSON: Structured data for API integration")
    print(f"  • PDF: Executive presentation format (requires additional libraries)")
    
    # Sample JSON report generation
    json_report = generator.generate_executive_report(
        attribution_data=attribution_data,
        performance_data=performance_data,
        report_type=ReportType.MONTHLY_PERFORMANCE,
        report_format=ReportFormat.JSON
    )
    
    # Parse and show JSON structure
    import json
    json_data = json.loads(json_report)
    
    print(f"\n🔧 JSON REPORT STRUCTURE:")
    print(f"  • Company: {json_data['company_name']}")
    print(f"  • Report Type: {json_data['report_type']}")
    print(f"  • Sections: {len(json_data['sections'])}")
    print(f"  • Strategic Recommendations: {len(json_data['strategic_recommendations'])}")
    print(f"  • Performance Data: {len(json_data['performance_data'])} channels")
    
    print("\n" + "="*60)
    print("🚀 Executive-grade marketing attribution reporting")
    print("💼 Board-ready insights and strategic recommendations")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_executive_reporting()