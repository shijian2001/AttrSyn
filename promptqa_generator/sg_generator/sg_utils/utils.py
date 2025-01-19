import networkx as nx
import sys

def make_and_description(names):
	names = [name for name in names]
	if len(names) == 0:
		return ""
	elif len(names) == 1:
		return names[0]
	elif len(names) == 2:
		return ' and '.join(names)
	else:
		names = names[:-1] + [f'and {names[-1]}']
		return ', '.join(names)

def normalized_sentence(sentence):
	return sentence.replace('_', ' ')

# add proposition
def mention_scene_attributes(type, scene_attribute):
    if type == "genre":
        return f"in the {scene_attribute} genre" 
    elif type == "artist":
        return f"in the style of the artist {scene_attribute}" 
    elif type == "painting style":
        return f"with the {scene_attribute} painting style" 
    elif type == "technique":
        return f"using the {scene_attribute} technique"
    elif type == "weather":
        return f"in {scene_attribute} weather"
    elif type == "location":
        return f"in {scene_attribute} scene"
    elif type == "lighting":
        return f"illuminated by {scene_attribute}"
    elif type == "size":
        return f"with {scene_attribute}"
    elif type == "view": 
        return f"viewed from {scene_attribute}"
    elif type == "depth of focus":
        return f"with {scene_attribute}"
    elif type == "focal length":
        return f"shot at {scene_attribute}"
    elif type == "camera model":
        return f"filmed with a {scene_attribute}"
    elif type == "camera movement":
        return f"filmed with {scene_attribute}"
    elif type == "camera gear":
        return f"using a {scene_attribute}"
    elif type == "video editting style":
        return f"edited in the {scene_attribute} style"
    elif type == "time span":
        return f"spanning {scene_attribute}"
    elif type == "ISO":
        return f"ISO {scene_attribute}"
    elif type == "aperture":
        return f"at {scene_attribute} aperture"
    else:
        return scene_attribute


def capitalize_first_char_if_letter(s):
	if len(s) == 0:
		return s
	if s[0].isalpha():
		return s[0].upper() + s[1:]
	return s


def get_element_num_dict(graph):
	object_nodes = [
		(n, d) for n, d in graph.nodes(data=True) if d["type"] == "object_node"
	]
	attribute_nodes = [
		(n, d) for n, d in graph.nodes(data=True) if d["type"] == "attribute_node"
	]
	relation_edges = [
		(n1, n2, d)
		for n1, n2, d in graph.edges(data=True)
		if d.get("type") == "relation_edge"
	]
	return {
		"object"   : len(object_nodes),
		"attribute": len(attribute_nodes),
		"relation" : len(relation_edges),
	}


# def convert_sg_to_json(graph: nx.DiGraph):
# 	nodes = list(graph.nodes(data=True))
# 	edges = list(graph.edges(data=True))
# 	graph = {
# 		"nodes": nodes,
# 		"edges": edges,
# 	}
# 	print(nodes)	
# 	print(edges)
# 	sys.exit()
# 	return graph

import networkx as nx

def convert_sg_to_json(scene_graph):
    # 提取节点和边
    nodes = list(scene_graph.nodes(data=True))
    edges = list(scene_graph.edges(data=True))
    print(nodes)
    print(edges)
    
    # 初始化结果字典
    result = {
        "labels": [],
        "attributes": [],
        "relations": []
    }
    
    # 记录每个节点的索引
    node_index_map = {}  # 节点名称到索引的映射
    object_nodes = []  # 记录 object_node 的索引
    attribute_dict = {}  # 存储各个 object_node 的属性
    
    # 处理节点
    for i, (node, attr) in enumerate(nodes):
        node_index_map[node] = i  # 建立节点名称到索引的映射
        if attr.get('type') == 'object_node':
            result["labels"].append(attr.get('value', ''))  # 填充 labels
            result["attributes"].append([])  # 初始化空的属性列表
            object_nodes.append(node)
        elif attr.get('type') == 'attribute_node':
            # 将属性添加到对应 object_node 的属性列表中
            parent_object = node.split('|')[1]  # 提取 object_id
            if parent_object not in attribute_dict:
                attribute_dict[parent_object] = []
            attribute_dict[parent_object].append(attr.get('value', ''))
    
    # 将属性填充到对应的 object_node
    for obj in object_nodes:
        obj_id = obj.split('_')[1]  # 从对象节点名称中提取 object_id
        obj_index = node_index_map[obj]
        result["attributes"][obj_index] = attribute_dict.get(obj_id, [])
    
    # 处理关系边，只保留 relation_edge
    for edge in edges:
        source, target, attr = edge
        if attr.get("type") == "relation_edge":
            relation_type = attr.get("value", "unknown_relation")
            source_index = node_index_map[source]
            target_index = node_index_map[target]
            result["relations"].append([source_index, relation_type, target_index])
    
    return result







def convert_json_to_sg(graph_json: dict):
	graph = nx.DiGraph()
	graph.add_nodes_from(graph_json["nodes"])
	graph.add_edges_from(graph_json["edges"])
	return graph

if __name__ == "__main__":
    # 这里是主程序执行时的代码
    scene_graph = {
    'nodes': [
        ('object_1', {'type': 'object_node', 'value': 'pier_table'}),
        ('object_2', {'type': 'object_node', 'value': 'nonvascular_organism'}),
        ('object_3', {'type': 'object_node', 'value': 'pastel blue'}),
        ('attribute|1|1', {'type': 'attribute_node', 'value': 'msu green'}),
        ('attribute|1|2', {'type': 'attribute_node', 'value': 'battleship grey'}),
        ('attribute|1|3', {'type': 'attribute_node', 'value': 'french sky blue'}),
        ('attribute|2|1', {'type': 'attribute_node', 'value': 'potted'}),
    ],
    'edges': []
}

    # 调用函数
    result = convert_sg_to_json(scene_graph)
    print(result)


