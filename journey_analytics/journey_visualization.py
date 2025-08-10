"""
Customer Journey Visualization System

Advanced visualization system for customer journeys, creating interactive
charts, network graphs, and executive dashboards to understand customer
behavior patterns and optimize marketing strategies.

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
from collections import defaultdict, Counter
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for journey visualizations."""
    width: int = 1200
    height: int = 800
    color_palette: List[str] = None
    theme: str = "light"  # light, dark, minimal
    show_labels: bool = True
    show_metrics: bool = True
    interactive: bool = True
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]


class JourneyVisualizer:
    """
    Advanced customer journey visualization system.
    
    Creates compelling visual representations of customer journeys including
    Sankey diagrams, network graphs, heatmaps, and executive dashboards.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize Journey Visualizer.
        
        Args:
            config: Visualization configuration settings
        """
        self.config = config or VisualizationConfig()
        
        # Visualization data storage
        self.journey_data = None
        self.journey_metrics = {}
        self.visualization_cache = {}
        
        logger.info("Journey visualization system initialized")
    
    def load_journey_data(self, journey_data: pd.DataFrame, 
                         journey_metrics: Optional[Dict[str, Any]] = None) -> 'JourneyVisualizer':
        """
        Load customer journey data for visualization.
        
        Args:
            journey_data: DataFrame with journey touchpoint data
            journey_metrics: Pre-calculated journey metrics
            
        Returns:
            Self for method chaining
        """
        self.journey_data = journey_data.copy()
        self.journey_metrics = journey_metrics or {}
        
        # Basic data preparation
        self.journey_data['timestamp'] = pd.to_datetime(self.journey_data['timestamp'])
        self.journey_data = self.journey_data.sort_values(['customer_id', 'timestamp'])
        
        logger.info(f"Loaded journey data: {len(self.journey_data)} touchpoints, "
                   f"{self.journey_data['customer_id'].nunique()} customers")
        
        return self
    
    def create_sankey_diagram(self, max_steps: int = 5) -> Dict[str, Any]:
        """
        Create Sankey diagram showing journey flow between channels.
        
        Args:
            max_steps: Maximum journey steps to include
            
        Returns:
            Sankey diagram data structure
        """
        if self.journey_data is None:
            raise ValueError("Journey data not loaded. Call load_journey_data() first.")
        
        # Build journey sequences
        sequences = []
        
        for customer_id, customer_data in self.journey_data.groupby('customer_id'):
            customer_data = customer_data.sort_values('timestamp')
            
            # Get channel sequence (limited to max_steps)
            channels = customer_data['channel'].tolist()[:max_steps]
            converted = customer_data['converted'].any()
            
            if len(channels) >= 2:  # Need at least 2 touchpoints
                sequences.append({
                    'channels': channels,
                    'converted': converted,
                    'revenue': customer_data['revenue'].sum()
                })
        
        # Build node and link data for Sankey
        nodes = []
        links = []
        
        # Create nodes (channel at each step)
        node_map = {}
        node_index = 0
        
        for step in range(max_steps):
            step_channels = set()
            for seq in sequences:
                if len(seq['channels']) > step:
                    step_channels.add(seq['channels'][step])
            
            for channel in step_channels:
                node_name = f"Step {step + 1}: {channel}"
                nodes.append({
                    'name': node_name,
                    'step': step,
                    'channel': channel
                })
                node_map[node_name] = node_index
                node_index += 1
        
        # Add conversion node
        conversion_node = "Conversion"
        nodes.append({'name': conversion_node, 'step': max_steps, 'channel': 'Conversion'})
        node_map[conversion_node] = node_index
        
        # Create links between steps
        link_counts = Counter()
        link_values = defaultdict(float)
        
        for seq in sequences:
            channels = seq['channels']
            converted = seq['converted']
            revenue = seq['revenue']
            
            # Links between consecutive steps
            for i in range(min(len(channels) - 1, max_steps - 1)):
                source_name = f"Step {i + 1}: {channels[i]}"
                target_name = f"Step {i + 2}: {channels[i + 1]}"
                
                if source_name in node_map and target_name in node_map:
                    link_key = (source_name, target_name)
                    link_counts[link_key] += 1
                    link_values[link_key] += revenue
            
            # Link to conversion
            if converted and len(channels) > 0:
                last_channel = channels[min(len(channels) - 1, max_steps - 1)]
                source_name = f"Step {min(len(channels), max_steps)}: {last_channel}"
                
                if source_name in node_map:
                    link_key = (source_name, conversion_node)
                    link_counts[link_key] += 1
                    link_values[link_key] += revenue
        
        # Build links array
        for (source_name, target_name), count in link_counts.items():
            if count >= 5:  # Minimum threshold
                links.append({
                    'source': node_map[source_name],
                    'target': node_map[target_name],
                    'value': count,
                    'revenue': link_values[(source_name, target_name)]
                })
        
        sankey_data = {
            'type': 'sankey',
            'title': 'Customer Journey Flow Analysis',
            'nodes': nodes,
            'links': links,
            'config': {
                'width': self.config.width,
                'height': self.config.height,
                'color_palette': self.config.color_palette
            },
            'metadata': {
                'total_sequences': len(sequences),
                'max_steps': max_steps,
                'unique_channels': len(set(self.journey_data['channel'])),
                'conversion_rate': sum(1 for seq in sequences if seq['converted']) / len(sequences) if sequences else 0
            }
        }
        
        self.visualization_cache['sankey'] = sankey_data
        logger.info(f"Created Sankey diagram with {len(nodes)} nodes and {len(links)} links")
        
        return sankey_data
    
    def create_network_graph(self) -> Dict[str, Any]:
        """
        Create network graph showing channel relationships and transitions.
        
        Returns:
            Network graph data structure
        """
        if self.journey_data is None:
            raise ValueError("Journey data not loaded. Call load_journey_data() first.")
        
        # Build channel transition network
        nodes = []
        edges = []
        
        # Calculate channel metrics
        channel_metrics = defaultdict(lambda: {
            'touchpoints': 0,
            'customers': set(),
            'conversions': 0,
            'revenue': 0.0,
            'avg_position': []
        })
        
        # Track transitions
        transitions = defaultdict(int)
        transition_revenues = defaultdict(float)
        
        for customer_id, customer_data in self.journey_data.groupby('customer_id'):
            customer_data = customer_data.sort_values('timestamp')
            channels = customer_data['channel'].tolist()
            converted = customer_data['converted'].any()
            total_revenue = customer_data['revenue'].sum()
            
            # Update channel metrics
            for i, (_, row) in enumerate(customer_data.iterrows()):
                channel = row['channel']
                channel_metrics[channel]['touchpoints'] += 1
                channel_metrics[channel]['customers'].add(customer_id)
                channel_metrics[channel]['revenue'] += row['revenue']
                channel_metrics[channel]['avg_position'].append(i + 1)
                
                if row['converted']:
                    channel_metrics[channel]['conversions'] += 1
            
            # Track transitions
            for i in range(len(channels) - 1):
                source = channels[i]
                target = channels[i + 1]
                
                if source != target:  # Skip self-loops
                    transitions[(source, target)] += 1
                    transition_revenues[(source, target)] += total_revenue
        
        # Create nodes (channels)
        for channel, metrics in channel_metrics.items():
            unique_customers = len(metrics['customers'])
            avg_position = np.mean(metrics['avg_position']) if metrics['avg_position'] else 0
            conversion_rate = metrics['conversions'] / unique_customers if unique_customers > 0 else 0
            
            nodes.append({
                'id': channel,
                'label': channel,
                'size': unique_customers,  # Node size based on customer count
                'touchpoints': metrics['touchpoints'],
                'customers': unique_customers,
                'conversions': metrics['conversions'],
                'conversion_rate': conversion_rate,
                'revenue': metrics['revenue'],
                'avg_position': avg_position,
                'color': self._get_channel_color(channel)
            })
        
        # Create edges (transitions)
        for (source, target), count in transitions.items():
            if count >= 3:  # Minimum threshold
                revenue = transition_revenues[(source, target)]
                
                edges.append({
                    'source': source,
                    'target': target,
                    'weight': count,
                    'revenue': revenue,
                    'width': min(count / 5, 10),  # Edge width based on frequency
                    'label': f"{count} journeys"
                })
        
        network_data = {
            'type': 'network',
            'title': 'Customer Journey Network',
            'nodes': nodes,
            'edges': edges,
            'config': {
                'width': self.config.width,
                'height': self.config.height,
                'color_palette': self.config.color_palette,
                'layout': 'force_directed'  # or 'circular', 'hierarchical'
            },
            'metadata': {
                'total_channels': len(nodes),
                'total_transitions': len(edges),
                'network_density': len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
            }
        }
        
        self.visualization_cache['network'] = network_data
        logger.info(f"Created network graph with {len(nodes)} nodes and {len(edges)} edges")
        
        return network_data
    
    def create_journey_heatmap(self, time_granularity: str = 'hour') -> Dict[str, Any]:
        """
        Create heatmap showing journey activity patterns over time.
        
        Args:
            time_granularity: Time granularity ('hour', 'day', 'week')
            
        Returns:
            Heatmap data structure
        """
        if self.journey_data is None:
            raise ValueError("Journey data not loaded. Call load_journey_data() first.")
        
        # Prepare time-based analysis
        data = self.journey_data.copy()
        
        if time_granularity == 'hour':
            data['time_period'] = data['timestamp'].dt.hour
            data['date_period'] = data['timestamp'].dt.date
            time_range = range(24)
            time_label = "Hour of Day"
        elif time_granularity == 'day':
            data['time_period'] = data['timestamp'].dt.day_name()
            data['date_period'] = data['timestamp'].dt.to_period('W')
            time_range = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            time_label = "Day of Week"
        else:  # week
            data['time_period'] = data['timestamp'].dt.to_period('W')
            data['date_period'] = data['timestamp'].dt.to_period('M')
            time_range = sorted(data['time_period'].unique())
            time_label = "Week"
        
        # Create heatmap matrix
        channels = sorted(data['channel'].unique())
        
        # Initialize matrices
        touchpoint_matrix = []
        conversion_matrix = []
        revenue_matrix = []
        
        for channel in channels:
            channel_data = data[data['channel'] == channel]
            
            touchpoint_row = []
            conversion_row = []
            revenue_row = []
            
            for time_period in time_range:
                period_data = channel_data[channel_data['time_period'] == time_period]
                
                touchpoints = len(period_data)
                conversions = period_data['converted'].sum()
                revenue = period_data['revenue'].sum()
                
                touchpoint_row.append(touchpoints)
                conversion_row.append(conversions)
                revenue_row.append(revenue)
            
            touchpoint_matrix.append(touchpoint_row)
            conversion_matrix.append(conversion_row)
            revenue_matrix.append(revenue_row)
        
        # Calculate intensity scores (normalized)
        max_touchpoints = max(max(row) for row in touchpoint_matrix) if touchpoint_matrix else 1
        max_conversions = max(max(row) for row in conversion_matrix) if conversion_matrix else 1
        max_revenue = max(max(row) for row in revenue_matrix) if revenue_matrix else 1
        
        normalized_touchpoints = [
            [val / max_touchpoints for val in row] 
            for row in touchpoint_matrix
        ]
        
        normalized_conversions = [
            [val / max_conversions for val in row] 
            for row in conversion_matrix
        ]
        
        normalized_revenue = [
            [val / max_revenue for val in row] 
            for row in revenue_matrix
        ]
        
        heatmap_data = {
            'type': 'heatmap',
            'title': f'Journey Activity Heatmap ({time_label})',
            'x_axis': {
                'title': time_label,
                'categories': [str(t) for t in time_range]
            },
            'y_axis': {
                'title': 'Marketing Channels',
                'categories': channels
            },
            'data_layers': {
                'touchpoints': {
                    'title': 'Touchpoint Volume',
                    'matrix': touchpoint_matrix,
                    'normalized': normalized_touchpoints,
                    'max_value': max_touchpoints
                },
                'conversions': {
                    'title': 'Conversion Count',
                    'matrix': conversion_matrix,
                    'normalized': normalized_conversions,
                    'max_value': max_conversions
                },
                'revenue': {
                    'title': 'Revenue Generated',
                    'matrix': revenue_matrix,
                    'normalized': normalized_revenue,
                    'max_value': max_revenue
                }
            },
            'config': {
                'width': self.config.width,
                'height': self.config.height,
                'color_scheme': 'viridis'  # or 'blues', 'reds', 'greens'
            },
            'metadata': {
                'time_granularity': time_granularity,
                'total_channels': len(channels),
                'time_periods': len(time_range)
            }
        }
        
        self.visualization_cache['heatmap'] = heatmap_data
        logger.info(f"Created heatmap with {len(channels)} channels and {len(time_range)} time periods")
        
        return heatmap_data
    
    def create_funnel_chart(self) -> Dict[str, Any]:
        """
        Create funnel chart showing conversion at each journey stage.
        
        Returns:
            Funnel chart data structure
        """
        if self.journey_data is None:
            raise ValueError("Journey data not loaded. Call load_journey_data() first.")
        
        # Define journey stages based on position and channel
        stage_mapping = {
            'awareness': ['Display', 'Video', 'Social'],
            'consideration': ['Search', 'Content', 'Email'],
            'intent': ['Product', 'Demo', 'Pricing'],
            'purchase': ['Direct', 'Checkout'],
            'retention': ['Support', 'Account', 'Usage']
        }
        
        # Assign stages to touchpoints
        data = self.journey_data.copy()
        data['stage'] = 'consideration'  # default
        
        for stage, channels in stage_mapping.items():
            for channel in channels:
                data.loc[data['channel'].str.contains(channel, case=False, na=False), 'stage'] = stage
        
        # Calculate funnel metrics by stage
        funnel_stages = []
        stage_order = ['awareness', 'consideration', 'intent', 'purchase', 'retention']
        
        for stage in stage_order:
            stage_data = data[data['stage'] == stage]
            
            if len(stage_data) == 0:
                continue
            
            customers_in_stage = stage_data['customer_id'].nunique()
            conversions = stage_data.groupby('customer_id')['converted'].any().sum()
            revenue = stage_data['revenue'].sum()
            
            conversion_rate = conversions / customers_in_stage if customers_in_stage > 0 else 0
            
            funnel_stages.append({
                'stage': stage.title(),
                'customers': customers_in_stage,
                'conversions': conversions,
                'conversion_rate': conversion_rate,
                'revenue': revenue,
                'avg_revenue_per_customer': revenue / customers_in_stage if customers_in_stage > 0 else 0
            })
        
        # Calculate drop-off rates
        for i in range(len(funnel_stages) - 1):
            current_customers = funnel_stages[i]['customers']
            next_customers = funnel_stages[i + 1]['customers']
            
            retention_rate = next_customers / current_customers if current_customers > 0 else 0
            dropoff_rate = 1 - retention_rate
            
            funnel_stages[i]['retention_rate'] = retention_rate
            funnel_stages[i]['dropoff_rate'] = dropoff_rate
            funnel_stages[i]['dropoff_count'] = current_customers - next_customers
        
        funnel_data = {
            'type': 'funnel',
            'title': 'Customer Journey Conversion Funnel',
            'stages': funnel_stages,
            'config': {
                'width': self.config.width,
                'height': self.config.height,
                'color_palette': self.config.color_palette,
                'show_percentages': True,
                'show_dropoff': True
            },
            'metadata': {
                'total_stages': len(funnel_stages),
                'overall_conversion_rate': funnel_stages[-1]['conversion_rate'] if funnel_stages else 0,
                'biggest_dropoff_stage': max(funnel_stages[:-1], 
                                           key=lambda x: x.get('dropoff_rate', 0))['stage'] if len(funnel_stages) > 1 else None
            }
        }
        
        self.visualization_cache['funnel'] = funnel_data
        logger.info(f"Created funnel chart with {len(funnel_stages)} stages")
        
        return funnel_data
    
    def create_cohort_analysis(self) -> Dict[str, Any]:
        """
        Create cohort analysis showing customer behavior over time.
        
        Returns:
            Cohort analysis data structure
        """
        if self.journey_data is None:
            raise ValueError("Journey data not loaded. Call load_journey_data() first.")
        
        # Define cohorts by first touchpoint month
        data = self.journey_data.copy()
        
        # Get first touchpoint for each customer
        first_touchpoints = data.groupby('customer_id')['timestamp'].min().reset_index()
        first_touchpoints['cohort_month'] = first_touchpoints['timestamp'].dt.to_period('M')
        
        # Merge cohort info back to data
        data = data.merge(first_touchpoints[['customer_id', 'cohort_month']], on='customer_id')
        
        # Calculate period numbers (months since first touchpoint)
        data['period_number'] = (
            data['timestamp'].dt.to_period('M') - data['cohort_month']
        ).apply(attrgetter('n'))
        
        # Build cohort table
        cohort_data = data.groupby(['cohort_month', 'period_number'])['customer_id'].nunique().reset_index()
        cohort_table = cohort_data.pivot(index='cohort_month', 
                                        columns='period_number', 
                                        values='customer_id').fillna(0)
        
        # Calculate cohort sizes
        cohort_sizes = data.groupby('cohort_month')['customer_id'].nunique()
        
        # Calculate retention rates
        retention_table = cohort_table.divide(cohort_sizes, axis=0)
        
        # Calculate revenue cohorts
        revenue_data = data.groupby(['cohort_month', 'period_number'])['revenue'].sum().reset_index()
        revenue_table = revenue_data.pivot(index='cohort_month', 
                                          columns='period_number', 
                                          values='revenue').fillna(0)
        
        # Convert to lists for visualization
        cohort_months = [str(month) for month in cohort_table.index]
        period_numbers = list(cohort_table.columns)
        
        retention_matrix = retention_table.values.tolist()
        revenue_matrix = revenue_table.values.tolist()
        
        cohort_analysis_data = {
            'type': 'cohort',
            'title': 'Customer Journey Cohort Analysis',
            'cohorts': cohort_months,
            'periods': period_numbers,
            'data_layers': {
                'retention': {
                    'title': 'Customer Retention Rate',
                    'matrix': retention_matrix,
                    'format': 'percentage'
                },
                'revenue': {
                    'title': 'Revenue per Cohort',
                    'matrix': revenue_matrix,
                    'format': 'currency'
                }
            },
            'config': {
                'width': self.config.width,
                'height': self.config.height,
                'color_scheme': 'blues'
            },
            'metadata': {
                'total_cohorts': len(cohort_months),
                'max_periods': len(period_numbers),
                'cohort_sizes': cohort_sizes.to_dict()
            }
        }
        
        self.visualization_cache['cohort'] = cohort_analysis_data
        logger.info(f"Created cohort analysis with {len(cohort_months)} cohorts")
        
        return cohort_analysis_data
    
    def create_executive_dashboard(self) -> Dict[str, Any]:
        """
        Create executive dashboard with key journey insights.
        
        Returns:
            Executive dashboard data structure
        """
        if self.journey_data is None:
            raise ValueError("Journey data not loaded. Call load_journey_data() first.")
        
        # Calculate key metrics
        total_customers = self.journey_data['customer_id'].nunique()
        total_touchpoints = len(self.journey_data)
        total_conversions = self.journey_data.groupby('customer_id')['converted'].any().sum()
        total_revenue = self.journey_data['revenue'].sum()
        
        conversion_rate = total_conversions / total_customers if total_customers > 0 else 0
        avg_touchpoints_per_customer = total_touchpoints / total_customers if total_customers > 0 else 0
        revenue_per_customer = total_revenue / total_customers if total_customers > 0 else 0
        revenue_per_conversion = total_revenue / total_conversions if total_conversions > 0 else 0
        
        # Channel performance
        channel_metrics = self.journey_data.groupby('channel').agg({
            'customer_id': 'nunique',
            'converted': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        channel_metrics['conversion_rate'] = (
            channel_metrics['converted'] / channel_metrics['customer_id']
        )
        channel_metrics['revenue_per_customer'] = (
            channel_metrics['revenue'] / channel_metrics['customer_id']
        )
        
        # Top performing channels
        top_channels = channel_metrics.nlargest(5, 'revenue')[
            ['channel', 'customer_id', 'conversion_rate', 'revenue']
        ].to_dict('records')
        
        # Journey length analysis
        journey_lengths = self.journey_data.groupby('customer_id').size()
        avg_journey_length = journey_lengths.mean()
        
        # Time-based analysis
        data_copy = self.journey_data.copy()
        data_copy['hour'] = data_copy['timestamp'].dt.hour
        data_copy['day_of_week'] = data_copy['timestamp'].dt.day_name()
        
        peak_hour = data_copy['hour'].value_counts().index[0]
        peak_day = data_copy['day_of_week'].value_counts().index[0]
        
        # Conversion trends (by day)
        daily_conversions = data_copy.groupby(data_copy['timestamp'].dt.date)['converted'].sum()
        conversion_trend = "increasing" if daily_conversions.iloc[-7:].mean() > daily_conversions.iloc[-14:-7].mean() else "decreasing"
        
        dashboard_data = {
            'type': 'dashboard',
            'title': 'Customer Journey Executive Dashboard',
            'summary_metrics': {
                'total_customers': total_customers,
                'total_touchpoints': total_touchpoints,
                'total_conversions': total_conversions,
                'total_revenue': total_revenue,
                'conversion_rate': conversion_rate,
                'avg_touchpoints_per_customer': avg_touchpoints_per_customer,
                'revenue_per_customer': revenue_per_customer,
                'revenue_per_conversion': revenue_per_conversion,
                'avg_journey_length': avg_journey_length
            },
            'performance_insights': {
                'top_channels': top_channels,
                'peak_activity_hour': peak_hour,
                'peak_activity_day': peak_day,
                'conversion_trend': conversion_trend,
                'highest_converting_channel': channel_metrics.loc[
                    channel_metrics['conversion_rate'].idxmax(), 'channel'
                ] if not channel_metrics.empty else None,
                'highest_revenue_channel': channel_metrics.loc[
                    channel_metrics['revenue'].idxmax(), 'channel'
                ] if not channel_metrics.empty else None
            },
            'visualizations': {
                'channel_performance_chart': {
                    'type': 'bar',
                    'data': channel_metrics[['channel', 'revenue']].to_dict('records'),
                    'x_axis': 'channel',
                    'y_axis': 'revenue',
                    'title': 'Revenue by Channel'
                },
                'conversion_rate_chart': {
                    'type': 'bar',
                    'data': channel_metrics[['channel', 'conversion_rate']].to_dict('records'),
                    'x_axis': 'channel',
                    'y_axis': 'conversion_rate',
                    'title': 'Conversion Rate by Channel'
                },
                'journey_length_distribution': {
                    'type': 'histogram',
                    'data': journey_lengths.tolist(),
                    'title': 'Journey Length Distribution',
                    'x_axis': 'Number of Touchpoints',
                    'y_axis': 'Number of Customers'
                },
                'hourly_activity': {
                    'type': 'line',
                    'data': data_copy['hour'].value_counts().sort_index().to_dict(),
                    'title': 'Touchpoint Activity by Hour',
                    'x_axis': 'Hour of Day',
                    'y_axis': 'Touchpoint Count'
                }
            },
            'config': {
                'width': self.config.width,
                'height': self.config.height,
                'color_palette': self.config.color_palette,
                'theme': self.config.theme
            },
            'metadata': {
                'data_period': {
                    'start': self.journey_data['timestamp'].min().isoformat(),
                    'end': self.journey_data['timestamp'].max().isoformat()
                },
                'unique_channels': self.journey_data['channel'].nunique(),
                'generated_at': datetime.now().isoformat()
            }
        }
        
        self.visualization_cache['dashboard'] = dashboard_data
        logger.info("Created executive dashboard with key journey metrics")
        
        return dashboard_data
    
    def _get_channel_color(self, channel: str) -> str:
        """Get color for a specific channel."""
        channel_colors = {
            'Search': '#1f77b4',
            'Display': '#ff7f0e', 
            'Social': '#2ca02c',
            'Email': '#d62728',
            'Direct': '#9467bd',
            'Referral': '#8c564b',
            'Video': '#e377c2',
            'Mobile': '#7f7f7f',
            'Affiliate': '#bcbd22'
        }
        
        return channel_colors.get(channel, self.config.color_palette[hash(channel) % len(self.config.color_palette)])
    
    def export_visualization(self, viz_type: str, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """
        Export visualization data in specified format.
        
        Args:
            viz_type: Type of visualization ('sankey', 'network', 'heatmap', 'funnel', 'cohort', 'dashboard')
            format: Export format ('json', 'html', 'svg')
            
        Returns:
            Exported visualization data
        """
        if viz_type not in self.visualization_cache:
            raise ValueError(f"Visualization '{viz_type}' not found. Create it first.")
        
        viz_data = self.visualization_cache[viz_type]
        
        if format == 'json':
            return json.dumps(viz_data, indent=2, default=str)
        elif format == 'html':
            return self._generate_html_visualization(viz_data)
        elif format == 'svg':
            return self._generate_svg_visualization(viz_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_html_visualization(self, viz_data: Dict[str, Any]) -> str:
        """Generate HTML representation of visualization."""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{viz_data.get('title', 'Customer Journey Visualization')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .viz-container {{ max-width: {viz_data.get('config', {}).get('width', 1200)}px; }}
                .metric {{ padding: 10px; margin: 5px; background: #f5f5f5; border-radius: 5px; }}
                .chart-placeholder {{ 
                    width: 100%; 
                    height: {viz_data.get('config', {}).get('height', 600)}px;
                    border: 2px dashed #ccc;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="viz-container">
                <h1>{viz_data.get('title', 'Visualization')}</h1>
                <div class="chart-placeholder">
                    Interactive {viz_data.get('type', 'chart').title()} Chart Would Appear Here
                    <br>
                    (Requires D3.js or similar visualization library)
                </div>
                <div class="metadata">
                    <h3>Visualization Details:</h3>
                    <pre>{json.dumps(viz_data.get('metadata', {}), indent=2)}</pre>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_svg_visualization(self, viz_data: Dict[str, Any]) -> str:
        """Generate SVG representation of visualization (simplified)."""
        
        width = viz_data.get('config', {}).get('width', 800)
        height = viz_data.get('config', {}).get('height', 600)
        
        svg_template = f"""
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="{width}" height="{height}" fill="white" stroke="#ccc" stroke-width="2"/>
            <text x="{width//2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">
                {viz_data.get('title', 'Visualization')}
            </text>
            <text x="{width//2}" y="{height//2}" text-anchor="middle" font-family="Arial" font-size="14" fill="#666">
                {viz_data.get('type', 'Chart').title()} Visualization
            </text>
            <text x="{width//2}" y="{height//2 + 20}" text-anchor="middle" font-family="Arial" font-size="12" fill="#999">
                (SVG implementation would render actual chart here)
            </text>
        </svg>
        """
        
        return svg_template
    
    def generate_visualization_report(self) -> str:
        """Generate comprehensive visualization report."""
        
        report = "# Customer Journey Visualization Report\n\n"
        report += "**Advanced Journey Visualizations by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        if self.journey_data is not None:
            # Data overview
            total_customers = self.journey_data['customer_id'].nunique()
            total_touchpoints = len(self.journey_data)
            
            report += f"## Data Overview\n\n"
            report += f"- **Total Customers**: {total_customers:,}\n"
            report += f"- **Total Touchpoints**: {total_touchpoints:,}\n"
            report += f"- **Unique Channels**: {self.journey_data['channel'].nunique()}\n"
            report += f"- **Date Range**: {self.journey_data['timestamp'].min().date()} to {self.journey_data['timestamp'].max().date()}\n\n"
        
        # Available visualizations
        report += f"## Available Visualizations\n\n"
        
        viz_descriptions = {
            'sankey': "**Sankey Diagram**: Shows flow of customers between channels at different journey stages",
            'network': "**Network Graph**: Displays channel relationships and transition patterns",
            'heatmap': "**Journey Heatmap**: Reveals activity patterns across time periods and channels",
            'funnel': "**Conversion Funnel**: Analyzes conversion rates at each journey stage",
            'cohort': "**Cohort Analysis**: Tracks customer behavior over time by acquisition cohort",
            'dashboard': "**Executive Dashboard**: Comprehensive overview with key metrics and insights"
        }
        
        for viz_type, description in viz_descriptions.items():
            status = "✅ Generated" if viz_type in self.visualization_cache else "⚪ Available"
            report += f"{status} {description}\n"
        
        report += "\n"
        
        # Generated visualizations summary
        if self.visualization_cache:
            report += f"## Generated Visualization Summary\n\n"
            
            for viz_type, viz_data in self.visualization_cache.items():
                metadata = viz_data.get('metadata', {})
                report += f"### {viz_data.get('title', viz_type.title())}\n"
                
                if viz_type == 'sankey':
                    report += f"- **Nodes**: {len(viz_data['nodes'])}\n"
                    report += f"- **Links**: {len(viz_data['links'])}\n"
                    report += f"- **Journey Sequences**: {metadata.get('total_sequences', 'N/A')}\n"
                
                elif viz_type == 'network':
                    report += f"- **Channels**: {len(viz_data['nodes'])}\n"
                    report += f"- **Transitions**: {len(viz_data['edges'])}\n"
                    report += f"- **Network Density**: {metadata.get('network_density', 0):.3f}\n"
                
                elif viz_type == 'heatmap':
                    report += f"- **Channels**: {metadata.get('total_channels', 'N/A')}\n"
                    report += f"- **Time Periods**: {metadata.get('time_periods', 'N/A')}\n"
                    report += f"- **Granularity**: {metadata.get('time_granularity', 'N/A')}\n"
                
                elif viz_type == 'funnel':
                    stages = len(viz_data['stages'])
                    conversion_rate = metadata.get('overall_conversion_rate', 0)
                    report += f"- **Funnel Stages**: {stages}\n"
                    report += f"- **Overall Conversion Rate**: {conversion_rate:.1%}\n"
                    
                    biggest_dropoff = metadata.get('biggest_dropoff_stage')
                    if biggest_dropoff:
                        report += f"- **Biggest Drop-off Stage**: {biggest_dropoff}\n"
                
                elif viz_type == 'cohort':
                    report += f"- **Cohorts**: {metadata.get('total_cohorts', 'N/A')}\n"
                    report += f"- **Max Periods**: {metadata.get('max_periods', 'N/A')}\n"
                
                elif viz_type == 'dashboard':
                    summary = viz_data.get('summary_metrics', {})
                    report += f"- **Total Revenue**: ${summary.get('total_revenue', 0):,.0f}\n"
                    report += f"- **Conversion Rate**: {summary.get('conversion_rate', 0):.1%}\n"
                    report += f"- **Avg Journey Length**: {summary.get('avg_journey_length', 0):.1f} touchpoints\n"
                
                report += "\n"
        
        # Visualization insights
        report += f"## Key Visualization Insights\n\n"
        
        if 'dashboard' in self.visualization_cache:
            dashboard_data = self.visualization_cache['dashboard']
            insights = dashboard_data.get('performance_insights', {})
            
            highest_converting = insights.get('highest_converting_channel')
            highest_revenue = insights.get('highest_revenue_channel')
            peak_hour = insights.get('peak_activity_hour')
            conversion_trend = insights.get('conversion_trend')
            
            if highest_converting:
                report += f"- **Best Converting Channel**: {highest_converting}\n"
            if highest_revenue:
                report += f"- **Highest Revenue Channel**: {highest_revenue}\n"
            if peak_hour:
                report += f"- **Peak Activity Hour**: {peak_hour}:00\n"
            if conversion_trend:
                report += f"- **Conversion Trend**: {conversion_trend.title()}\n"
        
        # Usage recommendations
        report += f"\n## Visualization Usage Recommendations\n\n"
        report += f"1. **Sankey Diagram**: Use for understanding customer flow patterns and identifying drop-off points\n"
        report += f"2. **Network Graph**: Ideal for analyzing channel relationships and planning cross-channel strategies\n"
        report += f"3. **Heatmap**: Perfect for timing optimization and resource allocation decisions\n"
        report += f"4. **Funnel Analysis**: Essential for conversion optimization and stage-specific improvements\n"
        report += f"5. **Cohort Analysis**: Valuable for long-term customer value and retention strategies\n"
        report += f"6. **Executive Dashboard**: Complete overview for strategic decision-making\n\n"
        
        report += "---\n*These visualizations provide comprehensive insights into customer journey behavior. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom visualization implementations.*"
        
        return report


def demo_journey_visualization():
    """Executive demonstration of Journey Visualization system."""
    
    print("=== Customer Journey Visualization: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Generate sample journey data
    customers = []
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct', 'Video']
    
    for customer_id in range(1, 301):
        # Random journey characteristics
        journey_length = np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.3, 0.3, 0.2, 0.1])
        conversion_prob = 0.18
        
        # Generate journey
        start_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 90))
        converted = np.random.random() < conversion_prob
        
        customer_channels = np.random.choice(channels, size=journey_length, replace=True)
        
        for i, channel in enumerate(customer_channels):
            days_offset = i * np.random.exponential(3)  # Average 3 days between touchpoints
            timestamp = start_date + timedelta(days=days_offset, hours=np.random.randint(0, 24))
            
            # Revenue only on conversion touchpoint
            revenue = 0.0
            is_converted = converted and (i == journey_length - 1)
            
            if is_converted:
                revenue = np.random.normal(400, 100)
                revenue = max(revenue, 100)  # Minimum revenue
            
            customers.append({
                'customer_id': f'customer_{customer_id}',
                'timestamp': timestamp,
                'channel': channel,
                'converted': is_converted,
                'revenue': revenue,
                'touchpoint_type': 'paid_search' if channel == 'Search' else 'display_ad'
            })
    
    journey_data = pd.DataFrame(customers)
    
    print(f"📊 Generated {len(journey_data)} touchpoints across {journey_data['customer_id'].nunique()} customers")
    print(f"📈 Overall conversion rate: {journey_data.groupby('customer_id')['converted'].any().mean():.1%}")
    
    # Initialize visualizer
    config = VisualizationConfig(
        width=1200,
        height=800,
        theme="light",
        interactive=True
    )
    
    visualizer = JourneyVisualizer(config=config)
    visualizer.load_journey_data(journey_data)
    
    print(f"\n🎨 Creating customer journey visualizations...")
    
    # Create various visualizations
    print("\n📊 JOURNEY VISUALIZATION RESULTS")
    print("=" * 50)
    
    # 1. Executive Dashboard
    dashboard = visualizer.create_executive_dashboard()
    summary = dashboard['summary_metrics']
    
    print(f"\n📈 Executive Dashboard Summary:")
    print(f"  • Total Customers: {summary['total_customers']:,}")
    print(f"  • Total Touchpoints: {summary['total_touchpoints']:,}")
    print(f"  • Conversion Rate: {summary['conversion_rate']:.1%}")
    print(f"  • Total Revenue: ${summary['total_revenue']:,.0f}")
    print(f"  • Revenue per Customer: ${summary['revenue_per_customer']:.0f}")
    print(f"  • Avg Journey Length: {summary['avg_journey_length']:.1f} touchpoints")
    
    # Top channels from dashboard
    insights = dashboard['performance_insights']
    print(f"\n🏆 Top Performing Channels:")
    for i, channel_data in enumerate(insights['top_channels'][:3], 1):
        print(f"  {i}. {channel_data['channel']}: ${channel_data['revenue']:,.0f} revenue, "
              f"{channel_data['conversion_rate']:.1%} conversion rate")
    
    # 2. Sankey Diagram
    sankey = visualizer.create_sankey_diagram(max_steps=4)
    sankey_meta = sankey['metadata']
    
    print(f"\n🌊 Sankey Flow Analysis:")
    print(f"  • Journey Sequences: {sankey_meta['total_sequences']:,}")
    print(f"  • Flow Nodes: {len(sankey['nodes'])}")
    print(f"  • Flow Links: {len(sankey['links'])}")
    print(f"  • Unique Channels: {sankey_meta['unique_channels']}")
    print(f"  • Sequence Conversion Rate: {sankey_meta['conversion_rate']:.1%}")
    
    # 3. Network Graph
    network = visualizer.create_network_graph()
    network_meta = network['metadata']
    
    print(f"\n🕸️ Channel Network Analysis:")
    print(f"  • Network Channels: {network_meta['total_channels']}")
    print(f"  • Channel Transitions: {network_meta['total_transitions']}")
    print(f"  • Network Density: {network_meta['network_density']:.3f}")
    
    # Show strongest connections
    strong_edges = sorted(network['edges'], key=lambda x: x['weight'], reverse=True)[:3]
    print(f"  • Top Transitions:")
    for edge in strong_edges:
        print(f"    - {edge['source']} → {edge['target']}: {edge['weight']} journeys")
    
    # 4. Journey Heatmap
    heatmap = visualizer.create_journey_heatmap(time_granularity='hour')
    heatmap_meta = heatmap['metadata']
    
    print(f"\n🗺️ Activity Heatmap Analysis:")
    print(f"  • Time Granularity: {heatmap_meta['time_granularity']}")
    print(f"  • Channels Analyzed: {heatmap_meta['total_channels']}")
    print(f"  • Time Periods: {heatmap_meta['time_periods']}")
    
    # Find peak activity
    touchpoint_data = heatmap['data_layers']['touchpoints']
    max_activity = max(max(row) for row in touchpoint_data['matrix'])
    print(f"  • Peak Activity: {max_activity} touchpoints in single hour/channel")
    
    # 5. Conversion Funnel
    funnel = visualizer.create_funnel_chart()
    funnel_meta = funnel['metadata']
    
    print(f"\n🚀 Conversion Funnel Analysis:")
    print(f"  • Funnel Stages: {funnel_meta['total_stages']}")
    print(f"  • Overall Conversion Rate: {funnel_meta['overall_conversion_rate']:.1%}")
    
    if funnel_meta['biggest_dropoff_stage']:
        print(f"  • Biggest Drop-off Stage: {funnel_meta['biggest_dropoff_stage']}")
    
    # Show funnel stages
    print(f"  • Stage Performance:")
    for stage in funnel['stages']:
        print(f"    - {stage['stage']}: {stage['customers']:,} customers → "
              f"{stage['conversions']} conversions ({stage['conversion_rate']:.1%})")
    
    # 6. Cohort Analysis
    cohort = visualizer.create_cohort_analysis()
    cohort_meta = cohort['metadata']
    
    print(f"\n👥 Cohort Analysis:")
    print(f"  • Total Cohorts: {cohort_meta['total_cohorts']}")
    print(f"  • Analysis Periods: {cohort_meta['max_periods']}")
    
    # Show cohort sizes
    cohort_sizes = cohort_meta['cohort_sizes']
    if cohort_sizes:
        largest_cohort = max(cohort_sizes, key=cohort_sizes.get)
        print(f"  • Largest Cohort: {largest_cohort} ({cohort_sizes[largest_cohort]} customers)")
    
    # Visualization export capabilities
    print(f"\n💾 Export Capabilities:")
    print(f"  • JSON: Structured data export for APIs")
    print(f"  • HTML: Interactive web-ready visualizations")  
    print(f"  • SVG: Scalable vector graphics for presentations")
    
    # Sample JSON export
    print(f"\n📄 Sample Visualization Export (Dashboard JSON):")
    json_export = visualizer.export_visualization('dashboard', 'json')
    
    # Show first 200 characters of JSON
    print(f"  {json_export[:200]}...")
    print(f"  [Total JSON length: {len(json_export)} characters]")
    
    # Performance insights
    print(f"\n💡 Key Visualization Insights:")
    
    # Channel insights
    top_channel = insights['top_channels'][0] if insights['top_channels'] else None
    if top_channel:
        print(f"  • {top_channel['channel']} is the top revenue driver with ${top_channel['revenue']:,.0f}")
    
    # Activity timing
    peak_hour = insights.get('peak_activity_hour')
    if peak_hour:
        print(f"  • Peak customer activity occurs at {peak_hour}:00")
    
    # Journey patterns
    avg_length = summary['avg_journey_length']
    if avg_length > 3:
        print(f"  • Complex journeys averaging {avg_length:.1f} touchpoints indicate need for nurturing")
    
    # Conversion trends
    trend = insights.get('conversion_trend')
    if trend:
        print(f"  • Conversion trend is currently {trend}")
    
    print("\n" + "="*60)
    print("🚀 Advanced customer journey visualization and insights")
    print("💼 Executive-grade visual analytics for strategic decisions")  
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


# Add attrgetter import for cohort analysis
try:
    from operator import attrgetter
except ImportError:
    # Fallback if attrgetter not available
    def attrgetter(attr):
        def getter(obj):
            return getattr(obj, attr)
        return getter


if __name__ == "__main__":
    demo_journey_visualization()