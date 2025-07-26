from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import re
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Change relative imports to absolute imports
from relation_extraction.semantic_relation_extractor import SemanticRelationExtractor
from llm.llm_client import LLMClient
from model_manager import ModelManager

class GraphBuilder:
    """
    Class for constructing a context-aware knowledge graph from extracted entities and relations.
    """

    def __init__(self,model_manager_instance: ModelManager):
        self.model_manager = model_manager_instance
        self.graph = nx.DiGraph()  # Use directed graph for relationships
        self.llm_client = LLMClient(model_manager_instance=self.model_manager)
        from relation_extraction.semantic_relation_extractor import SemanticRelationExtractor
        self.relation_extractor = SemanticRelationExtractor(model_manager_instance=self.model_manager)
        self.entity_nodes = {}
        self.relation_edges = {}
        
    def add_entity(self, entity: Dict[str, Any]) -> str:
        """
        Add an entity to the knowledge graph.

        Args:
            entity: A dictionary representing the entity to add.
            
        Returns:
            The ID of the added entity node
        """
        # Generate unique ID for the entity
        entity_id = self._generate_entity_id(entity)
        
        # Extract attributes for the node
        attributes = {
            # Use preferred_term or text for the label and name
            'label': entity.get('preferred_term', entity.get('text', '')),
            'name': entity.get('preferred_term', entity.get('text', '')),  # Add name attribute
            'type': entity.get('entity_type', 'UNKNOWN'),
            'original_text': entity.get('text', ''),
            'confidence': entity.get('confidence', 0.0)
        }
        
        # Add code information if available
        if 'code' in entity:
            attributes['code'] = entity['code']
            attributes['code_system'] = entity.get('code_system', 'UNKNOWN')
        
        # Add entity to graph
        self.graph.add_node(entity_id, **attributes)
        self.entity_nodes[entity_id] = attributes
        
        return entity_id

    def _generate_entity_id(self, entity: Dict[str, Any]) -> str:
        """Generate a unique ID for an entity"""
        entity_type = entity.get('entity_type', 'unknown').lower()
        entity_text = entity.get('text', '').lower().replace(' ', '_')
        # Sanitize the text for use in an ID
        entity_text = re.sub(r'[^\w_]', '', entity_text)
        # Truncate if too long
        if len(entity_text) > 30:
            entity_text = entity_text[:30]
        # Make sure we have something
        if not entity_text:
            entity_text = 'unnamed'
        # Return the ID
        return f"{entity_type}_{entity_text}"
        


    def add_relation(self, source: str, target: str, relation_type: str, 
                    confidence: float = 1.0, attributes: Dict[str, Any] = None) -> bool:
        """
        Add a relation between two entities in the knowledge graph.

        Args:
            source: The source entity ID
            target: The target entity ID
            relation_type: The type of relation
            confidence: Confidence score for the relation
            attributes: Additional attributes for the relation
            
        Returns:
            True if relation was added, False otherwise
        """
        if not source or not target:
            return False
            
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return False
            
        # Default attributes
        edge_attrs = {
            "relation": relation_type,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add custom attributes if provided
        if attributes:
            edge_attrs.update(attributes)
            
        # Add the edge with attributes
        self.graph.add_edge(source, target, **edge_attrs)
        return True

    def to_json(self, graph=None):
        """
        Convert the knowledge graph to a JSON serializable dictionary.
        
        Args:
            graph: Optional NetworkX graph to convert. If None, uses self.graph.
            
        Returns:
            Dictionary representation of the graph
        """
        # Use the provided graph or the internal one
        if graph is None:
            graph = self.graph
            
        # Create graph data structure
        graph_data = {
            "metadata": {
                "node_count": len(graph.nodes()),
                "edge_count": len(graph.edges()),
                "timestamp": datetime.now().isoformat(),
                "description": "Medical Knowledge Graph"
            },
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node, attrs in graph.nodes(data=True):
            # Convert non-serializable values to strings
            node_data = {"id": str(node)}
            for key, value in attrs.items():
                if isinstance(value, (str, int, float, bool)) and value is not None:
                    node_data[key] = value
                else:
                    try:
                        # Try to serialize complex objects
                        node_data[key] = json.dumps(value)
                    except:
                        # Fallback to string representation
                        node_data[key] = str(value)
            graph_data["nodes"].append(node_data)
        
        # Add edges
        for source, target, attrs in graph.edges(data=True):
            # Convert non-serializable values to strings
            edge_data = {
                "source": str(source),
                "target": str(target)
            }
            for key, value in attrs.items():
                if isinstance(value, (str, int, float, bool)) and value is not None:
                    edge_data[key] = value
                else:
                    try:
                        # Try to serialize complex objects
                        edge_data[key] = json.dumps(value)
                    except:
                        # Fallback to string representation
                        edge_data[key] = str(value)
            graph_data["edges"].append(edge_data)
        
        return graph_data
    
    def build_graph_from_entities_and_relations(self, entities, relations):
        """
        Build a knowledge graph from extracted entities and relations
        
        Args:
            entities: List of extracted entities
            relations: List of extracted relations
        
        Returns:
            NetworkX DiGraph representing the knowledge graph
        """
        import networkx as nx
        import re
        from datetime import datetime
        
        # Create a new directed graph
        self.graph = nx.DiGraph()
        
        # Add metadata to the graph
        self.graph.graph['name'] = 'Medical Knowledge Graph'
        self.graph.graph['description'] = 'Graph of medical entities and relationships'
        self.graph.graph['created'] = datetime.now().isoformat()
        
        # Track how many entities and relations we added
        entities_added = 0
        relations_added = 0
        
        # First, add all entities as nodes
        entity_map = {}  # Map to store entity text/id to node id
        
        print(f"Processing {len(entities)} entities for graph...")
        
        for entity in entities:
            try:
                # Generate a unique ID for this entity
                entity_id = self._generate_entity_id(entity)
                
                # Build attributes dictionary
                attributes = {}
                for key, value in entity.items():
                    if key not in ['id', 'confidence'] and value is not None:
                        attributes[key] = value
                
                # Add confidence if available
                if 'confidence' in entity:
                    attributes['confidence'] = entity['confidence']

                # Extract medical codes - make them first-class properties of the node
                if 'rxnorm_code' in entity:
                    attributes['rxnorm_code'] = entity['rxnorm_code']
                if 'icd10_code' in entity:
                    attributes['icd10_code'] = entity['icd10_code']
                if 'snomed_code' in entity:
                    attributes['snomed_code'] = entity['snomed_code']
                if 'loinc_code' in entity:
                    attributes['loinc_code'] = entity['loinc_code']
                    
                # Add the node with all entity attributes
                self.graph.add_node(entity_id, **attributes)
                entities_added += 1
                
                # Store in our map for quick lookup
                if 'text' in entity:
                    entity_map[entity['text'].lower()] = entity_id
                if 'id' in entity:
                    entity_map[entity['id']] = entity_id
            except Exception as e:
                print(f"Error adding entity {entity.get('text', 'unknown')}: {e}")
        
        print(f"Added {entities_added} entities as nodes")
        
        # Then add all relations as edges
        print(f"Processing {len(relations)} relations for graph...")
        
        for relation in relations:
            try:
                # Get source and target from relation
                source_text = relation.get('source', '')
                target_text = relation.get('target', '')
                
                # Skip if source or target is missing
                if not source_text or not target_text:
                    print(f"Skipping relation with missing source or target: {relation}")
                    continue
                    
                # Try to find the entities in our map
                source_node = None
                target_node = None
                
                # Try to match based on text
                if source_text.lower() in entity_map:
                    source_node = entity_map[source_text.lower()]
                
                if target_text.lower() in entity_map:
                    target_node = entity_map[target_text.lower()]
                
                # Fallback: try to match any text containing the source/target
                if not source_node:
                    for entity_text, node_id in entity_map.items():
                        if source_text.lower() in entity_text or entity_text in source_text.lower():
                            source_node = node_id
                            break
                
                if not target_node:
                    for entity_text, node_id in entity_map.items():
                        if target_text.lower() in entity_text or entity_text in target_text.lower():
                            target_node = node_id
                            break
                
                # If we found both source and target, add the edge
                if source_node and target_node:
                    # Create edge attributes
                    edge_attrs = {
                        'relation_type': relation.get('relation_type', 'RELATED_TO'),
                        'source_text': source_text,
                        'target_text': target_text,
                        'confidence': relation.get('confidence', 0.5),
                        'method': relation.get('method', 'unknown')
                    }
                    
                    # Add the edge to the graph
                    self.graph.add_edge(source_node, target_node, **edge_attrs)
                    relations_added += 1
                else:
                    print(f"Could not add relation - source: {source_text} ({source_node}), target: {target_text} ({target_node})")
            except Exception as e:
                print(f"Error adding relation {relation}: {e}")
        
        print(f"Added {relations_added} relations as edges")
        print(f"Built graph with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
        
        return self.graph

    def build_empty_graph(self):
        """
        Build an empty graph as a fallback when errors occur
        
        Returns:
            An empty NetworkX DiGraph
        """
        import networkx as nx
        from datetime import datetime
        
        self.graph = nx.DiGraph()
        self.graph.graph['name'] = 'Empty Medical Knowledge Graph'
        self.graph.graph['description'] = 'Fallback empty graph due to processing errors'
        self.graph.graph['created'] = datetime.now().isoformat()
        
        return self.graph

    def map_entities_if_needed(self, entities):
        """Map entities if needed using HybridEntityMapper"""
        # Import locally to avoid circular imports
        from entity_recognition.hybrid_entity_mapper import HybridEntityMapper
        mapper = HybridEntityMapper()
        return mapper.map_entities(entities)
    
    def process_transcript(self, transcript_data: Dict[str, Any]) -> nx.DiGraph:
        """
        Process a complete transcript to build a knowledge graph.
        
        Args:
            transcript_data: The transcript data with speaker turns
            
        Returns:
            The constructed knowledge graph
        """
        all_entities = []
        all_relations = []
        
        # Process each speaker turn
        for turn in transcript_data.get("speaker_turns", []):
            if "text" not in turn:
                continue
                
            # Extract entities from the turn
            turn_entities = self.relation_extractor.nlp.extract_entities(turn["text"])
            
            if turn_entities:
                # Map entities to standardized codes
                mapped_entities = self.map_entities_if_needed(turn_entities)
                all_entities.extend(mapped_entities)
                
                # Extract relations between entities
                speaker_id = turn.get("speaker_id")
                turn_relations = self.relation_extractor.extract_relations_with_context(
                    turn["text"], 
                    mapped_entities,
                    speaker_id=speaker_id
                )
                all_relations.extend(turn_relations)
        
        # Build the graph from all extracted entities and relations
        return self.build_graph_from_entities_and_relations(all_entities, all_relations)
    
    def refine_graph_with_llm(self) -> nx.DiGraph:
        """
        Use LLM to refine the knowledge graph by identifying missing connections
        and resolving contradictions.
        
        Returns:
            The refined knowledge graph
        """
        # Skip if graph is empty
        if not self.graph.nodes():
            return self.graph
        
        # Get all nodes and their information
        nodes_info = []
        for node_id, attrs in self.graph.nodes(data=True):
            nodes_info.append({
                "id": node_id,
                "label": attrs.get("label"),
                "text": attrs.get("text", node_id),
                "preferred_term": attrs.get("preferred_term", node_id)
            })
        
        # Get all edges and their information
        edges_info = []
        for source, target, attrs in self.graph.edges(data=True):
            edges_info.append({
                "source": source,
                "target": target,
                "relation": attrs.get("relation"),
                "confidence": attrs.get("confidence")
            })
        
        # Create prompt for LLM
        prompt = self._create_graph_refinement_prompt(nodes_info, edges_info)
        
        # Get suggestions from LLM
        suggestions = self.llm_client.generate_graph_refinement(prompt)
        
        # Apply suggestions to graph
        if suggestions:
            self._apply_llm_suggestions(suggestions)
        
        return self.graph
    
    def _create_graph_refinement_prompt(self, nodes_info: List[Dict], 
                                      edges_info: List[Dict]) -> str:
        """
        Create a prompt for LLM to refine the graph.
        
        Args:
            nodes_info: Information about graph nodes
            edges_info: Information about graph edges
            
        Returns:
            A prompt string for the LLM
        """
        prompt = "I have a medical knowledge graph with the following entities:\n\n"
        
        # Add entity information
        for node in nodes_info:
            prompt += f"- {node['text']} (Type: {node['label']})\n"
        
        prompt += "\nAnd the following relationships:\n\n"
        
        # Add relationship information
        for edge in edges_info:
            source = next((n['text'] for n in nodes_info if n['id'] == edge['source']), edge['source'])
            target = next((n['text'] for n in nodes_info if n['id'] == edge['target']), edge['target'])
            prompt += f"- {source} {edge['relation']} {target}\n"
        
        prompt += "\nPlease identify:\n"
        prompt += "1. Any missing relationships that should exist between these entities\n"
        prompt += "2. Any contradictions or incorrect relationships\n"
        prompt += "3. Any relationships that could be more specific\n\n"
        prompt += "Return your suggestions in JSON format with 'missing_relations', 'contradictions', and 'refinements' sections."
        
        return prompt
    
    def _apply_llm_suggestions(self, suggestions: Dict[str, Any]) -> None:
        """
        Apply suggestions from the LLM to refine the graph.
        
        Args:
            suggestions: Dictionary of suggestions from the LLM
        """
        try:
            # Add missing relations
            for relation in suggestions.get("missing_relations", []):
                source_text = relation.get("source")
                target_text = relation.get("target")
                relation_type = relation.get("relation")
                
                # Find the corresponding node IDs
                source_id = self._find_node_by_text(source_text)
                target_id = self._find_node_by_text(target_text)
                
                if source_id and target_id:
                    self.add_relation(
                        source_id, 
                        target_id, 
                        relation_type, 
                        confidence=0.8,  # LLM suggestion confidence
                        attributes={"source": "llm_suggestion"}
                    )
            
            # Remove contradictory relations
            for contradiction in suggestions.get("contradictions", []):
                source_text = contradiction.get("source")
                target_text = contradiction.get("target")
                
                source_id = self._find_node_by_text(source_text)
                target_id = self._find_node_by_text(target_text)
                
                if source_id and target_id and self.graph.has_edge(source_id, target_id):
                    self.graph.remove_edge(source_id, target_id)
            
            # Refine existing relations
            for refinement in suggestions.get("refinements", []):
                source_text = refinement.get("source")
                target_text = refinement.get("target")
                new_relation = refinement.get("new_relation")
                
                source_id = self._find_node_by_text(source_text)
                target_id = self._find_node_by_text(target_text)
                
                if source_id and target_id and self.graph.has_edge(source_id, target_id):
                    # Update the relation type
                    self.graph.edges[source_id, target_id]["relation"] = new_relation
                    self.graph.edges[source_id, target_id]["refined"] = True
        
        except Exception as e:
            print(f"Error applying LLM suggestions: {e}")
    
    def _find_node_by_text(self, text: str) -> Optional[str]:
        """
        Find a node ID by its text.
        
        Args:
            text: The text to search for
            
        Returns:
            The node ID if found, None otherwise
        """
        for node_id, attrs in self.graph.nodes(data=True):
            if (attrs.get("text", "").lower() == text.lower() or 
                attrs.get("preferred_term", "").lower() == text.lower()):
                return node_id
        return None
    
    def merge_graphs(self, other_graph: nx.DiGraph) -> nx.DiGraph:
        """
        Merge another graph into the current graph.
        
        Args:
            other_graph: The graph to merge
            
        Returns:
            The merged graph
        """
        # Create a mapping from other graph nodes to this graph nodes
        node_mapping = {}
        
        # First, add all nodes from the other graph that don't exist in this graph
        for node, attrs in other_graph.nodes(data=True):
            if node not in self.graph:
                self.graph.add_node(node, **attrs)
                node_mapping[node] = node
            else:
                # Node exists in both graphs, merge attributes
                for key, value in attrs.items():
                    if key not in self.graph.nodes[node]:
                        self.graph.nodes[node][key] = value
                node_mapping[node] = node
        
        # Then add all edges from the other graph
        for u, v, attrs in other_graph.edges(data=True):
            if node_mapping[u] in self.graph and node_mapping[v] in self.graph:
                if not self.graph.has_edge(node_mapping[u], node_mapping[v]):
                    self.graph.add_edge(node_mapping[u], node_mapping[v], **attrs)
                else:
                    # Edge exists in both graphs, merge attributes
                    for key, value in attrs.items():
                        if key not in self.graph.edges[node_mapping[u], node_mapping[v]]:
                            self.graph.edges[node_mapping[u], node_mapping[v]][key] = value
        
        return self.graph
    
    def export_graph_to_json(self, output_file: str) -> None:
        """
        Export the knowledge graph to a JSON file.
        
        Args:
            output_file: Path to the output file
        """
        # Create graph data structure
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node, attrs in self.graph.nodes(data=True):
            node_data = {"id": node}
            node_data.update(attrs)
            graph_data["nodes"].append(node_data)
        
        # Add edges
        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {
                "source": source,
                "target": target
            }
            edge_data.update(attrs)
            graph_data["edges"].append(edge_data)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
        print(f"Graph exported to {output_file} with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    
    def export_graph_to_graphml(self, output_file: str) -> None:
        """
        Export the knowledge graph to GraphML format.
        
        Args:
            output_file: Path to the output file
        """
        # Convert non-primitive attributes to strings
        for node, attrs in self.graph.nodes(data=True):
            for key, value in list(attrs.items()):
                if not isinstance(value, (str, int, float, bool)) or value is None:
                    attrs[key] = json.dumps(value)
        
        for source, target, attrs in self.graph.edges(data=True):
            for key, value in list(attrs.items()):
                if not isinstance(value, (str, int, float, bool)) or value is None:
                    attrs[key] = json.dumps(value)
        
        # Write to GraphML file
        nx.write_graphml(self.graph, output_file)
        print(f"Graph exported to GraphML: {output_file}")
    
    def visualize_graph(self, output_file: Optional[str] = None, 
                       show_plot: bool = False) -> None:
        """
        Visualize the knowledge graph.
        
        Args:
            output_file: Optional path to save the visualization
            show_plot: Whether to display the plot interactively
        """
        if not self.graph.nodes():
            print("Graph is empty, nothing to visualize")
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Define node colors based on entity types
        entity_types = set(attrs.get('label', 'UNKNOWN') 
                         for _, attrs in self.graph.nodes(data=True))
        color_map = {}
        colors = plt.cm.tab10(range(len(entity_types)))
        
        for i, entity_type in enumerate(entity_types):
            color_map[entity_type] = colors[i]
        
        # Get node colors
        node_colors = [color_map.get(attrs.get('label', 'UNKNOWN'), 'gray') 
                     for _, attrs in self.graph.nodes(data=True)]
        
        # Get edge colors based on relation type
        edge_labels = {(u, v): attrs.get('relation', '') 
                     for u, v, attrs in self.graph.edges(data=True)}
        
        # Create positions
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color=node_colors, 
                             node_size=300)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.7)
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos, 
            labels={node: attrs.get('text', node)[:15] + '...' if len(attrs.get('text', node)) > 15 else attrs.get('text', node) 
                   for node, attrs in self.graph.nodes(data=True)},
            font_size=8
        )
        
        # Draw edge labels (only for small graphs)
        if len(self.graph.edges()) < 50:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        
        # Add legend
        for entity_type, color in color_map.items():
            plt.plot([0], [0], color=color, label=entity_type)
        
        plt.legend(title="Entity Types", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Medical Knowledge Graph")
        plt.axis('off')
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {output_file}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()