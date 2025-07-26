from typing import Dict, List, Any, Tuple
import networkx as nx
import json
from llm.llm_client import LLMClient
import sys
import os
from model_manager import ModelManager

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class GraphValidator:
    """
    Class for validating the constructed knowledge graph and ensuring data integrity.
    """

    def __init__(self, model_manager_instance: ModelManager, graph=None):
        """
        Initialize the GraphValidator
        
        Args:
            graph: The knowledge graph to validate
        """
        self.model_manager= model_manager_instance
        self.graph = graph or nx.DiGraph()
        self.llm_client = LLMClient(model_manager_instance=self.model_manager)
        self.validation_issues = []


    def validate_graph(self, model_manager_instance: ModelManager, graph):
        """
        Validate the knowledge graph
        
        Args:
            graph: NetworkX graph to validate
            
        Returns:
            Dictionary with validation results
        """
        self.graph = graph
        
        # Initialize results
        results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check if graph is empty
        if len(graph.nodes()) == 0:
            results["warnings"].append("Graph is empty with no nodes")
            return results
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            results["warnings"].append(f"Graph contains {len(isolated_nodes)} isolated nodes")
        
        # Check connectivity only if there are multiple nodes
        if len(graph.nodes()) > 1:
            # Convert to undirected for connectivity check
            undirected = graph.to_undirected()
            if not nx.is_connected(undirected):
                results["warnings"].append("Graph is not fully connected")
                # Count connected components
                connected_components = list(nx.connected_components(undirected))
                results["warnings"].append(f"Graph has {len(connected_components)} connected components")
        
        # Check node attributes
        for node, attrs in graph.nodes(data=True):
            if 'entity_type' not in attrs:
                results["warnings"].append(f"Node {node} missing entity_type attribute")
        
        # Check edge attributes
        for source, target, attrs in graph.edges(data=True):
            if 'relation_type' not in attrs:
                results["warnings"].append(f"Edge {source}->{target} missing relation_type attribute")
        
        return results
    
    def _validate_node_properties(self, validation_results):
        """Validate that all nodes have required properties"""
        required_properties = ["type", "name"]
        
        for node, data in self.graph.nodes(data=True):
            for prop in required_properties:
                if prop not in data:
                    validation_results["errors"].append(f"Node {node} missing required property: {prop}")
                    validation_results["valid"] = False
    
    def _validate_edge_properties(self, validation_results):
        """Validate that all edges have required properties"""
        required_properties = ["type", "confidence"]
        
        for source, target, data in self.graph.edges(data=True):
            for prop in required_properties:
                if prop not in data:
                    validation_results["errors"].append(f"Edge {source}->{target} missing required property: {prop}")
                    validation_results["valid"] = False
                    
    def export_validation_report(self, output_path="graph_validation_report.json"):
        """
        Export a validation report to a JSON file
        
        Args:
            output_path: Path to save the validation report
            
        Returns:
            bool: True if report was exported successfully
        """
        if self.graph is None:
            return False
            
        validation_results = self.validate_graph()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting validation report: {e}")
            return False

    def set_graph(self, graph: nx.Graph) -> None:
        """
        Set the graph to validate.
        
        Args:
            graph: The knowledge graph to validate
        """
        self.graph = graph
        self.validation_issues = []

    def validate_structure(self) -> bool:
        """
        Validate the structure of the knowledge graph.
        
        Returns:
            bool: True if the structure is valid, False otherwise.
        """
        # Check if the graph has nodes
        if not self.graph.nodes():
            self._add_issue("Graph has no nodes")
            return False
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            self._add_issue(f"Found {len(isolated_nodes)} isolated nodes", 
                          {"isolated_nodes": isolated_nodes[:10]})  # Show first 10 only
        
        # Check for connectivity in undirected view
        if not nx.is_connected(self.graph.to_undirected()) and len(self.graph.nodes()) > 1:
            components = list(nx.connected_components(self.graph.to_undirected()))
            self._add_issue(f"Graph is not connected. Found {len(components)} components",
                          {"component_sizes": [len(c) for c in components]})
        
        # Check for cycles in the graph if it's a DAG (medical knowledge often should be)
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                self._add_issue(f"Found {len(cycles)} cycles in the graph", 
                              {"example_cycle": str(cycles[0])})
        except Exception:
            pass  # Some NetworkX algorithms only work on simple graphs

        return len(self.validation_issues) == 0

    def validate_data_integrity(self) -> bool:
        """
        Validate the data integrity of the knowledge graph.
        
        Returns:
            bool: True if data integrity is valid, False otherwise.
        """
        # Check all nodes have required attributes
        required_node_attrs = ["label", "text"]
        nodes_missing_attrs = []
        
        for node in self.graph.nodes():
            attrs = self.graph.nodes[node]
            for req_attr in required_node_attrs:
                if req_attr not in attrs:
                    nodes_missing_attrs.append((node, req_attr))
        
        if nodes_missing_attrs:
            self._add_issue(f"{len(nodes_missing_attrs)} nodes missing required attributes",
                          {"examples": nodes_missing_attrs[:5]})
        
        # Check all edges have relation type
        edges_missing_relation = []
        for u, v, attrs in self.graph.edges(data=True):
            if "relation" not in attrs:
                edges_missing_relation.append((u, v))
        
        if edges_missing_relation:
            self._add_issue(f"{len(edges_missing_relation)} edges missing relation type",
                          {"examples": edges_missing_relation[:5]})
        
        # Check for duplicate nodes (same entity with different IDs)
        duplicate_entities = self._find_duplicate_entities()
        if duplicate_entities:
            self._add_issue(f"Found {len(duplicate_entities)} potential duplicate entities",
                          {"examples": duplicate_entities[:5]})
        
        return len(self.validation_issues) == 0

    def validate_medical_consistency(self) -> bool:
        """
        Validate the medical consistency of the knowledge graph.
        
        Returns:
            bool: True if medically consistent, False otherwise.
        """
        # Check for contradictory relationships
        contradictions = self._find_contradictions()
        if contradictions:
            self._add_issue(f"Found {len(contradictions)} contradictory relationships",
                          {"examples": contradictions[:5]})
        
        # Use LLM to check for medical inconsistencies
        llm_inconsistencies = self._check_with_llm()
        if llm_inconsistencies:
            self._add_issue(f"LLM found {len(llm_inconsistencies)} medical inconsistencies",
                          {"issues": llm_inconsistencies[:5]})
        
        return len(self.validation_issues) == 0

    def run_validation(self) -> bool:
        """
        Run all validation checks on the knowledge graph.
        
        Returns:
            bool: True if all validations pass, False otherwise.
        """
        # Clear previous issues
        self.validation_issues = []
        
        # Run all validations
        structure_valid = self.validate_structure()
        integrity_valid = self.validate_data_integrity()
        consistency_valid = self.validate_medical_consistency()
        
        return structure_valid and integrity_valid and consistency_valid

    def report_validation_results(self) -> Dict[str, Any]:
        """
        Report the results of the validation checks.
        
        Returns:
            Dictionary with validation results
        """
        report = {
            "passed": len(self.validation_issues) == 0,
            "issues_count": len(self.validation_issues),
            "issues": self.validation_issues,
            "graph_stats": {
                "nodes": len(self.graph.nodes()),
                "edges": len(self.graph.edges()),
                "entity_types": self._count_entity_types(),
                "relation_types": self._count_relation_types()
            }
        }
        
        if report["passed"]:
            print("Knowledge graph validation passed.")
        else:
            print(f"Knowledge graph validation failed with {len(self.validation_issues)} issues.")
            
        return report

    def suggest_improvements(self) -> List[Dict[str, Any]]:
        """
        Suggest improvements for the knowledge graph.
        
        Returns:
            List of suggested improvements
        """
        suggestions = []
        
        # Suggest merging duplicate entities
        duplicates = self._find_duplicate_entities()
        if duplicates:
            for dup in duplicates:
                suggestions.append({
                    "type": "merge_entities",
                    "entities": dup,
                    "reason": "Potential duplicate entities detected"
                })
        
        # Suggest adding edges to connect isolated components
        if not nx.is_connected(self.graph.to_undirected()) and len(self.graph.nodes()) > 1:
            components = list(nx.connected_components(self.graph.to_undirected()))
            if len(components) > 1:
                for i in range(len(components) - 1):
                    comp1 = list(components[i])
                    comp2 = list(components[i+1])
                    
                    suggestions.append({
                        "type": "connect_components",
                        "component1": str(comp1[:3]) + "..." if len(comp1) > 3 else str(comp1),
                        "component2": str(comp2[:3]) + "..." if len(comp2) > 3 else str(comp2),
                        "reason": "Disconnected components detected"
                    })
        
        # Use LLM to suggest additional improvements
        llm_suggestions = self._get_llm_suggestions()
        suggestions.extend(llm_suggestions)
        
        return suggestions
    
    def _find_duplicate_entities(self) -> List[Tuple[str, str]]:
        """
        Find potential duplicate entities in the graph.
        
        Returns:
            List of tuples of potentially duplicate node IDs
        """
        duplicates = []
        nodes_by_text = {}
        nodes_by_code = {}
        
        # Group nodes by text or codes
        for node_id, attrs in self.graph.nodes(data=True):
            # Group by text
            text = attrs.get("text", "").lower()
            if text:
                if text not in nodes_by_text:
                    nodes_by_text[text] = []
                nodes_by_text[text].append(node_id)
            
            # Group by codes if available
            codes = attrs.get("codes", {})
            for system, code in codes.items():
                if code:
                    code_key = f"{system}:{code}"
                    if code_key not in nodes_by_code:
                        nodes_by_code[code_key] = []
                    nodes_by_code[code_key].append(node_id)
        
        # Find duplicates by text
        for text, node_ids in nodes_by_text.items():
            if len(node_ids) > 1:
                for i in range(len(node_ids)):
                    for j in range(i+1, len(node_ids)):
                        duplicates.append((node_ids[i], node_ids[j]))
        
        # Find duplicates by code
        for code, node_ids in nodes_by_code.items():
            if len(node_ids) > 1:
                for i in range(len(node_ids)):
                    for j in range(i+1, len(node_ids)):
                        duplicates.append((node_ids[i], node_ids[j]))
        
        return duplicates
    
    def _find_contradictions(self) -> List[Dict[str, Any]]:
        """
        Find contradictory relationships in the graph.
        
        Returns:
            List of contradictory relationships
        """
        contradictions = []
        
        # Define contradictory relation pairs
        contradictory_relations = {
            "TREATS": ["CAUSES", "WORSENS"],
            "CAUSES": ["TREATS", "PREVENTS"],
            "PREVENTS": ["CAUSES"],
            "IMPROVES": ["WORSENS"],
            "WORSENS": ["IMPROVES", "TREATS"]
        }
        
        # Check for contradictions
        for u, v, attrs1 in self.graph.edges(data=True):
            relation1 = attrs1.get("relation")
            if relation1 and relation1 in contradictory_relations:
                # Check for reverse edge with contradictory relation
                if self.graph.has_edge(v, u):
                    attrs2 = self.graph.edges[v, u]
                    relation2 = attrs2.get("relation")
                    
                    if relation2 in contradictory_relations[relation1]:
                        contradictions.append({
                            "entity1": u,
                            "entity2": v,
                            "relation1": relation1,
                            "relation2": relation2,
                            "text1": self.graph.nodes[u].get("text", u),
                            "text2": self.graph.nodes[v].get("text", v)
                        })
        
        return contradictions
    
    def _check_with_llm(self) -> List[Dict[str, Any]]:
        """
        Use LLM to check for medical inconsistencies.
        
        Returns:
            List of issues found by the LLM
        """
        # Only check if there are a reasonable number of nodes
        if not self.graph.nodes() or len(self.graph.nodes()) > 100:
            return []
        
        try:
            # Create a list of all nodes with their attributes
            nodes_info = []
            for node, attrs in self.graph.nodes(data=True):
                nodes_info.append({
                    "id": node,
                    "text": attrs.get("text", node),
                    "type": attrs.get("label", "UNKNOWN")
                })
            
            # Create a list of all relationships
            edges_info = []
            for u, v, attrs in self.graph.edges(data=True):
                edges_info.append({
                    "source": self.graph.nodes[u].get("text", u),
                    "target": self.graph.nodes[v].get("text", v),
                    "relation": attrs.get("relation", "RELATED_TO")
                })
            
            # Create prompt for LLM
            prompt = self._create_validation_prompt(nodes_info, edges_info)
            
            # Get validation results from LLM
            results = self.llm_client.validate_medical_knowledge(prompt)
            
            # Extract issues
            issues = []
            if isinstance(results, dict) and "issues" in results:
                issues = results["issues"]
            
            return issues
        
        except Exception as e:
            print(f"Error in LLM validation: {e}")
            return []
    
    def _get_llm_suggestions(self) -> List[Dict[str, Any]]:
        """
        Use LLM to suggest improvements to the graph.
        
        Returns:
            List of suggested improvements
        """
        # Only check if there are a reasonable number of nodes
        if not self.graph.nodes() or len(self.graph.nodes()) > 100:
            return []
        
        try:
            # Create a list of all nodes with their attributes
            nodes_info = []
            for node, attrs in self.graph.nodes(data=True):
                nodes_info.append({
                    "id": node,
                    "text": attrs.get("text", node),
                    "type": attrs.get("label", "UNKNOWN")
                })
            
            # Create a list of all relationships
            edges_info = []
            for u, v, attrs in self.graph.edges(data=True):
                edges_info.append({
                    "source": self.graph.nodes[u].get("text", u),
                    "target": self.graph.nodes[v].get("text", v),
                    "relation": attrs.get("relation", "RELATED_TO")
                })
            
            # Create prompt for LLM
            prompt = self._create_suggestion_prompt(nodes_info, edges_info)
            
            # Get suggestions from LLM
            suggestions = self.llm_client.suggest_graph_improvements(prompt)
            
            return suggestions if isinstance(suggestions, list) else []
        
        except Exception as e:
            print(f"Error in LLM suggestions: {e}")
            return []
    
    def _create_validation_prompt(self, nodes_info: List[Dict], edges_info: List[Dict]) -> str:
        """
        Create a prompt for LLM to validate the medical knowledge graph.
        
        Args:
            nodes_info: Information about graph nodes
            edges_info: Information about graph edges
            
        Returns:
            A prompt string for the LLM
        """
        prompt = "I need you to validate the medical consistency of this knowledge graph.\n\n"
        
        # Add entity information
        prompt += "Entities:\n"
        for node in nodes_info:
            prompt += f"- {node['text']} (Type: {node['type']})\n"
        
        prompt += "\nRelationships:\n"
        for edge in edges_info:
            prompt += f"- {edge['source']} {edge['relation']} {edge['target']}\n"
        
        prompt += "\nPlease identify any medical inconsistencies, contradictions, or factually incorrect"
        prompt += " statements in this knowledge graph. Return your findings as a JSON list under an 'issues' key,"
        prompt += " where each issue has 'description' and 'severity' (low/medium/high) fields."
        
        return prompt
    
    def _create_suggestion_prompt(self, nodes_info: List[Dict], edges_info: List[Dict]) -> str:
        """
        Create a prompt for LLM to suggest improvements to the graph.
        
        Args:
            nodes_info: Information about graph nodes
            edges_info: Information about graph edges
            
        Returns:
            A prompt string for the LLM
        """
        prompt = "I need suggestions to improve this medical knowledge graph.\n\n"
        
        # Add entity information
        prompt += "Entities:\n"
        for node in nodes_info:
            prompt += f"- {node['text']} (Type: {node['type']})\n"
        
        prompt += "\nRelationships:\n"
        for edge in edges_info:
            prompt += f"- {edge['source']} {edge['relation']} {edge['target']}\n"
        
        prompt += "\nPlease suggest improvements such as:\n"
        prompt += "1. Additional relationships that should exist\n"
        prompt += "2. Missing entities that would complete the graph\n"
        prompt += "3. Refinements to existing relationships\n"
        prompt += "4. Entity types that could be more specific\n\n"
        prompt += "Return your suggestions as a JSON list where each item has 'type', 'description', and 'reason' fields."
        
        return prompt
    
    def _count_entity_types(self) -> Dict[str, int]:
        """
        Count the number of entities of each type.
        
        Returns:
            Dictionary mapping entity types to counts
        """
        entity_counts = {}
        for _, attrs in self.graph.nodes(data=True):
            entity_type = attrs.get("label", "UNKNOWN")
            if entity_type not in entity_counts:
                entity_counts[entity_type] = 0
            entity_counts[entity_type] += 1
        
        return entity_counts
    
    def _count_relation_types(self) -> Dict[str, int]:
        """
        Count the number of relationships of each type.
        
        Returns:
            Dictionary mapping relation types to counts
        """
        relation_counts = {}
        for _, _, attrs in self.graph.edges(data=True):
            relation_type = attrs.get("relation", "UNKNOWN")
            if relation_type not in relation_counts:
                relation_counts[relation_type] = 0
            relation_counts[relation_type] += 1
        
        return relation_counts
    
    def _add_issue(self, message: str, details: Dict[str, Any] = None) -> None:
        """
        Add an issue to the validation issues list.
        
        Args:
            message: Issue message
            details: Optional issue details
        """
        issue = {
            "message": message,
            "details": details or {}
        }
        self.validation_issues.append(issue)