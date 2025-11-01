"""
PXL Postprocessor - UIP Step 2 Component
=======================================

Output formatting and result transformation for PXL (Philosophically Extended Logic) analysis.
Handles report generation, visualization data preparation, and integration with downstream systems.

Integrates with: PXL relation mapper, consistency checker, schema definitions, V2 framework protocols
Dependencies: Jinja2 templates, matplotlib/plotly for visualizations, JSON/XML/CSV output formats
"""

import base64
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from xml.dom import minidom

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from jinja2 import Environment, FileSystemLoader, Template
from LOGOS_AI.Advanced_Reasoning_Protocol.system_utillities.system_imports import *

# Import PXL schema components
try:
    from .pxl_schema import (
        ConsistencyReport,
        ConsistencyViolation,
        ModalProperties,
        PXLAnalysisConfig,
        PXLAnalysisResult,
        PXLConsistencyLevel,
        PXLRelation,
        PXLRelationType,
        TrinityVector,
        ValidationResult,
    )
except ImportError:
    # Fallback imports for development
    from LOGOS_AI.Advanced_Reasoning_Protocol.mathematical_foundations.pxl.pxl_schema import (
        ConsistencyReport,
        ConsistencyViolation,
        ModalProperties,
        PXLAnalysisConfig,
        PXLAnalysisResult,
        PXLConsistencyLevel,
        PXLRelation,
        PXLRelationType,
        TrinityVector,
        ValidationResult,
    )


class OutputFormat(Enum):
    """Supported output formats"""

    JSON = "json"
    XML = "xml"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    LATEX = "latex"
    YAML = "yaml"


class VisualizationType(Enum):
    """Types of visualizations"""

    RELATION_NETWORK = "relation_network"
    TRINITY_HEATMAP = "trinity_heatmap"
    CONSISTENCY_DASHBOARD = "consistency_dashboard"
    CONCEPT_HIERARCHY = "concept_hierarchy"
    MODAL_ANALYSIS = "modal_analysis"
    VIOLATION_SUMMARY = "violation_summary"


class ReportStyle(Enum):
    """Report styling options"""

    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    ACADEMIC = "academic"
    MINIMAL = "minimal"
    DETAILED = "detailed"


@dataclass
class PostprocessingConfig:
    """Configuration for PXL postprocessing"""

    output_format: OutputFormat = OutputFormat.JSON
    include_visualizations: bool = True
    visualization_types: List[VisualizationType] = field(
        default_factory=lambda: [
            VisualizationType.RELATION_NETWORK,
            VisualizationType.CONSISTENCY_DASHBOARD,
        ]
    )
    report_style: ReportStyle = ReportStyle.TECHNICAL

    # Output options
    pretty_print: bool = True
    include_metadata: bool = True
    include_raw_data: bool = False

    # Visualization options
    figure_size: Tuple[int, int] = (12, 8)
    color_scheme: str = "default"
    interactive_plots: bool = False

    # Filtering options
    min_relation_strength: float = 0.1
    max_violations_displayed: int = 50
    include_warnings: bool = True

    # Template options
    custom_template_path: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedOutput:
    """Result of postprocessing operation"""

    format_type: OutputFormat
    content: str
    metadata: Dict[str, Any]
    visualizations: Dict[str, Any] = field(default_factory=dict)
    file_size: int = 0
    processing_time: float = 0.0

    def save_to_file(self, filepath: str):
        """Save processed output to file"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.content)

    def get_content_summary(self) -> str:
        """Get summary of content"""
        return (
            f"{self.format_type.value.upper()} output ({len(self.content)} characters)"
        )


class RelationNetworkVisualizer:
    """Visualizer for PXL relation networks"""

    def __init__(self, config: PostprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_network_visualization(
        self,
        relations: List[PXLRelation],
        trinity_vectors: Optional[Dict[str, TrinityVector]] = None,
    ) -> str:
        """Create network visualization of PXL relations"""

        try:
            # Create NetworkX graph
            G = nx.DiGraph()

            # Add nodes and edges
            for relation in relations:
                if relation.strength >= self.config.min_relation_strength:
                    G.add_edge(
                        relation.source_concept,
                        relation.target_concept,
                        weight=relation.strength,
                        relation_type=relation.relation_type.value,
                        confidence=relation.confidence,
                    )

            if len(G.nodes()) == 0:
                return self._create_empty_network_message()

            # Create matplotlib figure
            plt.figure(figsize=self.config.figure_size)

            # Layout
            if len(G.nodes()) < 50:
                pos = nx.spring_layout(G, k=2, iterations=50)
            else:
                pos = nx.kamada_kawai_layout(G)

            # Node colors based on Trinity vectors
            node_colors = []
            if trinity_vectors:
                for node in G.nodes():
                    if node in trinity_vectors:
                        trinity_vec = trinity_vectors[node]
                        # Use Trinity magnitude as color intensity
                        intensity = trinity_vec.magnitude() / np.sqrt(3)  # Normalize
                        node_colors.append(plt.cm.viridis(intensity))
                    else:
                        node_colors.append("lightgray")
            else:
                node_colors = ["lightblue"] * len(G.nodes())

            # Edge colors based on relation strength
            edge_colors = [G[u][v]["weight"] for u, v in G.edges()]

            # Draw network
            nx.draw_networkx_nodes(
                G, pos, node_color=node_colors, node_size=300, alpha=0.8
            )
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

            edges = nx.draw_networkx_edges(
                G,
                pos,
                edge_color=edge_colors,
                edge_cmap=plt.cm.Blues,
                width=2,
                alpha=0.7,
                arrows=True,
                arrowsize=20,
            )

            # Add colorbar for edge weights
            if edges:
                plt.colorbar(edges, label="Relation Strength")

            plt.title("PXL Relation Network", fontsize=14, fontweight="bold")
            plt.axis("off")
            plt.tight_layout()

            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            self.logger.error(f"Network visualization failed: {e}")
            return self._create_error_visualization("Network visualization error")

    def _create_empty_network_message(self) -> str:
        """Create message for empty network"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        ax.text(
            0.5,
            0.5,
            "No relations meet minimum strength threshold",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def _create_error_visualization(self, error_msg: str) -> str:
        """Create error visualization"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        ax.text(
            0.5,
            0.5,
            f"Error: {error_msg}",
            ha="center",
            va="center",
            fontsize=12,
            color="red",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"


class TrinityHeatmapVisualizer:
    """Visualizer for Trinity vector heatmaps"""

    def __init__(self, config: PostprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_trinity_heatmap(self, trinity_vectors: Dict[str, TrinityVector]) -> str:
        """Create heatmap of Trinity vectors"""

        try:
            if not trinity_vectors:
                return self._create_empty_heatmap_message()

            # Prepare data
            concepts = list(trinity_vectors.keys())[
                :20
            ]  # Limit to prevent overcrowding
            dimensions = ["Essence", "Generation", "Temporal"]

            # Create matrix
            matrix = np.zeros((len(concepts), 3))
            for i, concept in enumerate(concepts):
                trinity_vec = trinity_vectors[concept]
                matrix[i] = [
                    trinity_vec.essence,
                    trinity_vec.generation,
                    trinity_vec.temporal,
                ]

            # Create heatmap
            fig, ax = plt.subplots(figsize=self.config.figure_size)

            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

            # Set ticks and labels
            ax.set_xticks(range(len(dimensions)))
            ax.set_xticklabels(dimensions)
            ax.set_yticks(range(len(concepts)))
            ax.set_yticklabels(concepts, fontsize=8)

            # Add text annotations
            for i in range(len(concepts)):
                for j in range(len(dimensions)):
                    text = ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

            ax.set_title("Trinity Vector Heatmap", fontsize=14, fontweight="bold")
            plt.colorbar(im, label="Trinity Dimension Value")
            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            self.logger.error(f"Trinity heatmap visualization failed: {e}")
            return self._create_error_visualization("Trinity heatmap error")

    def _create_empty_heatmap_message(self) -> str:
        """Create message for empty heatmap"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        ax.text(
            0.5,
            0.5,
            "No Trinity vectors available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def _create_error_visualization(self, error_msg: str) -> str:
        """Create error visualization"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        ax.text(
            0.5,
            0.5,
            f"Error: {error_msg}",
            ha="center",
            va="center",
            fontsize=12,
            color="red",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"


class ConsistencyDashboardVisualizer:
    """Visualizer for consistency analysis dashboard"""

    def __init__(self, config: PostprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_consistency_dashboard(
        self, consistency_report: ConsistencyReport
    ) -> str:
        """Create consistency dashboard visualization"""

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2, figsize=self.config.figure_size
            )

            # 1. Violation types pie chart
            if consistency_report.violations_by_type:
                violation_types = list(consistency_report.violations_by_type.keys())
                violation_counts = [
                    len(violations)
                    for violations in consistency_report.violations_by_type.values()
                ]

                ax1.pie(
                    violation_counts,
                    labels=violation_types,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax1.set_title("Violations by Type")
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "No violations found",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )
                ax1.set_title("Violations by Type")

            # 2. Severity distribution
            severity_labels = ["Critical", "Major", "Moderate", "Minor", "Warning"]
            severity_counts = []

            for level in [
                PXLConsistencyLevel.CRITICAL_VIOLATION,
                PXLConsistencyLevel.MAJOR_INCONSISTENCY,
                PXLConsistencyLevel.MODERATE_TENSION,
                PXLConsistencyLevel.MINOR_DISCREPANCY,
            ]:
                count = len(consistency_report.violations_by_severity.get(level, []))
                severity_counts.append(count)

            # Add warnings (assuming they exist in some form)
            severity_counts.append(0)  # Placeholder for warnings

            colors = ["red", "orange", "yellow", "lightblue", "green"]
            ax2.bar(severity_labels, severity_counts, color=colors)
            ax2.set_title("Violations by Severity")
            ax2.tick_params(axis="x", rotation=45)

            # 3. Consistency scores
            if consistency_report.local_consistency_scores:
                concepts = list(consistency_report.local_consistency_scores.keys())[
                    :10
                ]  # Top 10
                scores = [
                    consistency_report.local_consistency_scores[concept]
                    for concept in concepts
                ]

                ax3.barh(concepts, scores, color="skyblue")
                ax3.set_title("Local Consistency Scores")
                ax3.set_xlabel("Consistency Score")
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No local scores available",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                )
                ax3.set_title("Local Consistency Scores")

            # 4. Global consistency gauge
            global_score = consistency_report.global_consistency_score

            # Create gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)

            ax4.plot(theta, r, "k-", linewidth=2)
            ax4.fill_between(theta, 0, r, alpha=0.3, color="lightgray")

            # Color-coded sections
            critical_end = np.pi * 0.2
            major_end = np.pi * 0.4
            moderate_end = np.pi * 0.6
            minor_end = np.pi * 0.8

            if global_score <= 0.2:
                color = "red"
            elif global_score <= 0.4:
                color = "orange"
            elif global_score <= 0.6:
                color = "yellow"
            elif global_score <= 0.8:
                color = "lightgreen"
            else:
                color = "green"

            # Draw indicator
            score_angle = np.pi * (1 - global_score)
            ax4.arrow(
                0,
                0,
                np.cos(score_angle),
                np.sin(score_angle),
                head_width=0.05,
                head_length=0.1,
                fc=color,
                ec=color,
                linewidth=3,
            )

            ax4.set_xlim(-1.2, 1.2)
            ax4.set_ylim(0, 1.2)
            ax4.set_aspect("equal")
            ax4.axis("off")
            ax4.text(
                0,
                -0.2,
                f"Global Score: {global_score:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
            ax4.set_title("Global Consistency")

            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            self.logger.error(f"Consistency dashboard visualization failed: {e}")
            return self._create_error_visualization("Consistency dashboard error")

    def _create_error_visualization(self, error_msg: str) -> str:
        """Create error visualization"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        ax.text(
            0.5,
            0.5,
            f"Error: {error_msg}",
            ha="center",
            va="center",
            fontsize=12,
            color="red",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"


class OutputFormatter:
    """Handles different output format generation"""

    def __init__(self, config: PostprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def format_to_json(self, analysis_result: PXLAnalysisResult) -> str:
        """Format analysis result to JSON"""

        def serialize_object(obj):
            """Custom serializer for complex objects"""
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (set, tuple)):
                return list(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj)

        # Convert to serializable dictionary
        data = {
            "analysis_metadata": {
                "processing_time": analysis_result.processing_time,
                "timestamp": datetime.now().isoformat(),
                "config": asdict(analysis_result.analysis_config),
            },
            "relations": [asdict(relation) for relation in analysis_result.relations],
            "consistency_report": {
                "total_violations": analysis_result.consistency_report.total_violations,
                "global_consistency_score": analysis_result.consistency_report.global_consistency_score,
                "violations_by_type": {
                    str(vtype): [asdict(v) for v in violations]
                    for vtype, violations in analysis_result.consistency_report.violations_by_type.items()
                },
                "local_consistency_scores": analysis_result.consistency_report.local_consistency_scores,
                "resolution_recommendations": analysis_result.consistency_report.resolution_recommendations,
            },
            "trinity_analysis": {
                concept: asdict(vector)
                for concept, vector in analysis_result.trinity_analysis.items()
            },
            "modal_analysis": {
                concept: asdict(props)
                for concept, props in analysis_result.modal_analysis.items()
            },
            "semantic_clusters": analysis_result.semantic_clusters,
        }

        if self.config.include_metadata:
            data["metadata"] = analysis_result.metadata

        if self.config.pretty_print:
            return json.dumps(
                data, indent=2, default=serialize_object, ensure_ascii=False
            )
        else:
            return json.dumps(data, default=serialize_object, ensure_ascii=False)

    def format_to_xml(self, analysis_result: PXLAnalysisResult) -> str:
        """Format analysis result to XML"""

        root = ET.Element("pxl_analysis_result")

        # Metadata
        metadata_elem = ET.SubElement(root, "metadata")
        ET.SubElement(metadata_elem, "processing_time").text = str(
            analysis_result.processing_time
        )
        ET.SubElement(metadata_elem, "timestamp").text = datetime.now().isoformat()

        # Relations
        relations_elem = ET.SubElement(root, "relations")
        for relation in analysis_result.relations:
            rel_elem = ET.SubElement(relations_elem, "relation")
            ET.SubElement(rel_elem, "source_concept").text = relation.source_concept
            ET.SubElement(rel_elem, "target_concept").text = relation.target_concept
            ET.SubElement(rel_elem, "relation_type").text = relation.relation_type.value
            ET.SubElement(rel_elem, "strength").text = str(relation.strength)
            ET.SubElement(rel_elem, "confidence").text = str(relation.confidence)

        # Consistency report
        consistency_elem = ET.SubElement(root, "consistency_report")
        ET.SubElement(consistency_elem, "total_violations").text = str(
            analysis_result.consistency_report.total_violations
        )
        ET.SubElement(consistency_elem, "global_consistency_score").text = str(
            analysis_result.consistency_report.global_consistency_score
        )

        # Pretty print XML
        rough_string = ET.tostring(root, "unicode")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def format_to_csv(self, analysis_result: PXLAnalysisResult) -> str:
        """Format analysis result to CSV (relations only)"""

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "source_concept",
                "target_concept",
                "relation_type",
                "strength",
                "confidence",
                "trinity_coherence",
                "modal_necessity",
            ]
        )

        # Relations data
        for relation in analysis_result.relations:
            writer.writerow(
                [
                    relation.source_concept,
                    relation.target_concept,
                    relation.relation_type.value,
                    relation.strength,
                    relation.confidence,
                    relation.trinity_coherence or "",
                    relation.modal_necessity or "",
                ]
            )

        return output.getvalue()

    def format_to_markdown(self, analysis_result: PXLAnalysisResult) -> str:
        """Format analysis result to Markdown"""

        md_content = []

        # Title and metadata
        md_content.append("# PXL Analysis Report")
        md_content.append("")
        md_content.append(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        md_content.append(
            f"**Processing Time:** {analysis_result.processing_time:.2f} seconds"
        )
        md_content.append("")

        # Executive Summary
        md_content.append("## Executive Summary")
        md_content.append("")
        consistency_score = analysis_result.consistency_report.global_consistency_score
        md_content.append(f"- **Relations Analyzed:** {len(analysis_result.relations)}")
        md_content.append(f"- **Global Consistency Score:** {consistency_score:.2f}")
        md_content.append(
            f"- **Total Violations:** {analysis_result.consistency_report.total_violations}"
        )
        md_content.append("")

        # Consistency Analysis
        if analysis_result.consistency_report.total_violations > 0:
            md_content.append("## Consistency Analysis")
            md_content.append("")

            # Violations by type
            md_content.append("### Violations by Type")
            md_content.append("")
            for (
                vtype,
                violations,
            ) in analysis_result.consistency_report.violations_by_type.items():
                md_content.append(f"- **{vtype}:** {len(violations)} violations")
            md_content.append("")

            # Resolution recommendations
            if analysis_result.consistency_report.resolution_recommendations:
                md_content.append("### Resolution Recommendations")
                md_content.append("")
                for i, recommendation in enumerate(
                    analysis_result.consistency_report.resolution_recommendations, 1
                ):
                    md_content.append(f"{i}. {recommendation}")
                md_content.append("")

        # Trinity Analysis
        if analysis_result.trinity_analysis:
            md_content.append("## Trinity Analysis")
            md_content.append("")
            md_content.append("| Concept | Essence | Generation | Temporal |")
            md_content.append("|---------|---------|------------|----------|")

            for concept, vector in list(analysis_result.trinity_analysis.items())[
                :10
            ]:  # Top 10
                md_content.append(
                    f"| {concept} | {vector.essence:.3f} | {vector.generation:.3f} | {vector.temporal:.3f} |"
                )
            md_content.append("")

        # Relations Summary
        md_content.append("## Relations Summary")
        md_content.append("")

        # Group relations by type
        relations_by_type = {}
        for relation in analysis_result.relations:
            rel_type = relation.relation_type.value
            if rel_type not in relations_by_type:
                relations_by_type[rel_type] = []
            relations_by_type[rel_type].append(relation)

        for rel_type, relations in relations_by_type.items():
            md_content.append(
                f"### {rel_type.replace('_', ' ').title()} ({len(relations)} relations)"
            )
            md_content.append("")

            # Show top relations
            sorted_relations = sorted(relations, key=lambda r: r.strength, reverse=True)
            for relation in sorted_relations[:5]:  # Top 5
                md_content.append(
                    f"- {relation.source_concept} → {relation.target_concept} (strength: {relation.strength:.2f})"
                )

            if len(relations) > 5:
                md_content.append(f"- ... and {len(relations) - 5} more")
            md_content.append("")

        return "\n".join(md_content)


class ReportTemplateEngine:
    """Template engine for custom report formats"""

    def __init__(self, config: PostprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Setup Jinja2 environment
        if (
            self.config.custom_template_path
            and Path(self.config.custom_template_path).exists()
        ):
            loader = FileSystemLoader(
                str(Path(self.config.custom_template_path).parent)
            )
            self.env = Environment(loader=loader)
        else:
            # Use built-in templates
            self.env = Environment(loader=FileSystemLoader("."))

    def render_template(
        self, analysis_result: PXLAnalysisResult, template_name: str = "default"
    ) -> str:
        """Render analysis result using template"""

        try:
            # Prepare template context
            context = {
                "result": analysis_result,
                "timestamp": datetime.now(),
                "config": self.config,
                **self.config.template_variables,
            }

            # Built-in templates
            if template_name == "default" or not self.config.custom_template_path:
                return self._render_default_template(context)

            # Custom template
            template = self.env.get_template(template_name)
            return template.render(**context)

        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}")
            return f"Template rendering error: {e}"

    def _render_default_template(self, context: Dict[str, Any]) -> str:
        """Render default HTML template"""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PXL Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { border-bottom: 2px solid #333; padding-bottom: 20px; }
                .summary { background-color: #f5f5f5; padding: 20px; margin: 20px 0; }
                .section { margin: 30px 0; }
                .violation { background-color: #ffe6e6; padding: 10px; margin: 10px 0; border-left: 4px solid #ff0000; }
                .relation { background-color: #e6f3ff; padding: 10px; margin: 10px 0; border-left: 4px solid #0066cc; }
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>PXL Analysis Report</h1>
                <p><strong>Generated:</strong> {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                <p><strong>Processing Time:</strong> {{ result.processing_time|round(2) }} seconds</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <ul>
                    <li><strong>Relations Analyzed:</strong> {{ result.relations|length }}</li>
                    <li><strong>Global Consistency Score:</strong> {{ result.consistency_report.global_consistency_score|round(3) }}</li>
                    <li><strong>Total Violations:</strong> {{ result.consistency_report.total_violations }}</li>
                    <li><strong>Trinity Concepts:</strong> {{ result.trinity_analysis|length }}</li>
                </ul>
            </div>
            
            {% if result.consistency_report.total_violations > 0 %}
            <div class="section">
                <h2>Consistency Violations</h2>
                {% for violation_type, violations in result.consistency_report.violations_by_type.items() %}
                    <h3>{{ violation_type|replace('_', ' ')|title }} ({{ violations|length }})</h3>
                    {% for violation in violations[:5] %}
                        <div class="violation">
                            <strong>{{ violation.severity.value|title }}:</strong> {{ violation.description }}
                            {% if violation.suggested_resolution %}
                                <br><em>Resolution:</em> {{ violation.suggested_resolution }}
                            {% endif %}
                        </div>
                    {% endfor %}
                {% endfor %}
            </div>
            {% endif %}
            
            <div class="section">
                <h2>Relation Analysis</h2>
                <table>
                    <tr>
                        <th>Source</th>
                        <th>Target</th>
                        <th>Type</th>
                        <th>Strength</th>
                        <th>Confidence</th>
                    </tr>
                    {% for relation in result.relations[:20] %}
                    <tr>
                        <td>{{ relation.source_concept }}</td>
                        <td>{{ relation.target_concept }}</td>
                        <td>{{ relation.relation_type.value|replace('_', ' ')|title }}</td>
                        <td>{{ relation.strength|round(3) }}</td>
                        <td>{{ relation.confidence|round(3) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        """

        template = Template(html_template)
        return template.render(**context)


class PXLPostprocessor:
    """Main PXL postprocessing engine"""

    def __init__(self, config: Optional[PostprocessingConfig] = None):
        self.config = config or PostprocessingConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.network_visualizer = RelationNetworkVisualizer(self.config)
        self.trinity_visualizer = TrinityHeatmapVisualizer(self.config)
        self.consistency_visualizer = ConsistencyDashboardVisualizer(self.config)
        self.formatter = OutputFormatter(self.config)
        self.template_engine = ReportTemplateEngine(self.config)

        self.logger.info("PXL postprocessor initialized")

    def process_analysis_result(
        self, analysis_result: PXLAnalysisResult
    ) -> ProcessedOutput:
        """
        Process PXL analysis result into formatted output

        Args:
            analysis_result: Result from PXL analysis

        Returns:
            ProcessedOutput: Formatted output with visualizations
        """

        start_time = time.time()

        try:
            # Generate visualizations
            visualizations = {}

            if self.config.include_visualizations:
                visualizations = self._generate_visualizations(analysis_result)

            # Format content based on output format
            if self.config.output_format == OutputFormat.JSON:
                content = self.formatter.format_to_json(analysis_result)
            elif self.config.output_format == OutputFormat.XML:
                content = self.formatter.format_to_xml(analysis_result)
            elif self.config.output_format == OutputFormat.CSV:
                content = self.formatter.format_to_csv(analysis_result)
            elif self.config.output_format == OutputFormat.MARKDOWN:
                content = self.formatter.format_to_markdown(analysis_result)
            elif self.config.output_format == OutputFormat.HTML:
                content = self.template_engine.render_template(
                    analysis_result, "default"
                )
            else:
                # Fallback to JSON
                content = self.formatter.format_to_json(analysis_result)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create processed output
            processed_output = ProcessedOutput(
                format_type=self.config.output_format,
                content=content,
                metadata={
                    "relations_processed": len(analysis_result.relations),
                    "violations_found": analysis_result.consistency_report.total_violations,
                    "global_consistency": analysis_result.consistency_report.global_consistency_score,
                    "trinity_concepts": len(analysis_result.trinity_analysis),
                    "modal_concepts": len(analysis_result.modal_analysis),
                    "visualizations_generated": len(visualizations),
                },
                visualizations=visualizations,
                file_size=len(content.encode("utf-8")),
                processing_time=processing_time,
            )

            self.logger.info(f"Postprocessing completed in {processing_time:.2f}s")
            return processed_output

        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")
            raise

    def _generate_visualizations(
        self, analysis_result: PXLAnalysisResult
    ) -> Dict[str, Any]:
        """Generate all requested visualizations"""

        visualizations = {}

        try:
            if VisualizationType.RELATION_NETWORK in self.config.visualization_types:
                visualizations["relation_network"] = (
                    self.network_visualizer.create_network_visualization(
                        analysis_result.relations, analysis_result.trinity_analysis
                    )
                )

            if VisualizationType.TRINITY_HEATMAP in self.config.visualization_types:
                if analysis_result.trinity_analysis:
                    visualizations["trinity_heatmap"] = (
                        self.trinity_visualizer.create_trinity_heatmap(
                            analysis_result.trinity_analysis
                        )
                    )

            if (
                VisualizationType.CONSISTENCY_DASHBOARD
                in self.config.visualization_types
            ):
                visualizations["consistency_dashboard"] = (
                    self.consistency_visualizer.create_consistency_dashboard(
                        analysis_result.consistency_report
                    )
                )

            self.logger.debug(f"Generated {len(visualizations)} visualizations")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")

        return visualizations

    def generate_report(
        self, analysis_result: PXLAnalysisResult, format_type: str = "json"
    ) -> str:
        """
        Generate formatted report

        Args:
            analysis_result: PXL analysis result
            format_type: Output format type

        Returns:
            str: Formatted report content
        """

        # Temporarily override format
        original_format = self.config.output_format
        try:
            self.config.output_format = OutputFormat(format_type.lower())
            processed_output = self.process_analysis_result(analysis_result)
            return processed_output.content
        finally:
            self.config.output_format = original_format


# Global postprocessor instance with default configuration
pxl_postprocessor = PXLPostprocessor()


__all__ = [
    "OutputFormat",
    "VisualizationType",
    "ReportStyle",
    "PostprocessingConfig",
    "ProcessedOutput",
    "RelationNetworkVisualizer",
    "TrinityHeatmapVisualizer",
    "ConsistencyDashboardVisualizer",
    "OutputFormatter",
    "ReportTemplateEngine",
    "PXLPostprocessor",
    "pxl_postprocessor",
]
