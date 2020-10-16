import re
from os import path
from collections import defaultdict
from pprint import pprint

 

KG_DIR = 'KGfacts'
FACTS = 'KGfacts-CloseWorld.csv'
ENTITIES = 'Qid-NamedEntityMapping.csv'

Q_PATTERN = re.compile('Q\d+')


def is_q_id(s):
    return Q_PATTERN.match(s) is not None
 

def load_entities(data_dir):
    denotationals = {}
    entities = {}
    for line in open(path.join(data_dir, KG_DIR, ENTITIES), encoding="utf-8"):
        tokens = line.split('\t')
        if len(tokens) < 2: continue
        qid, denotational = tokens
        denotational = denotational[2:-2].encode().decode('unicode-escape').encode('latin1').decode('utf-8')
     
        assert denotational not in denotationals
        denotationals[denotational] = qid        

        assert qid not in entities
        entities[qid] = denotational
    return denotationals, entities


NO_COMMA, COMMA = 0, 1
def load_kg(data_dir):
    denotationals, entities = load_entities(data_dir)    

    kg = defaultdict(lambda: defaultdict(set))
    for line in open(path.join(data_dir, KG_DIR, FACTS), encoding="utf-8"):
        tokens = []
        buffer, state = [], NO_COMMA
        for c in line:
            if c == '\n':
                tokens.append(''.join(buffer))
                break
            buffer.append(c)
            if state == NO_COMMA:
                if c == ',': state = COMMA
            elif state == COMMA:
                if c == ' ': state = NO_COMMA
                else:
                    tokens.append(''.join(buffer[:-2]))
                    state = NO_COMMA
                    buffer = [c]
        if len(tokens) != 3: continue        

        sbj, rel, obj = tokens
        if sbj in denotationals: sbj = denotationals[sbj]
        if obj in denotationals: obj = denotationals[obj]
        kg[sbj][rel].add(obj)
        if is_q_id(sbj) and sbj in entities:
            kg[sbj]['name'].add(entities[sbj])
    return kg, denotationals, entities

def fact_extractor(kg, root, entities):
    relations = []
    next_hops = []
    for relation, entity_set in kg[root].items():
        for entity in entity_set:
            if Q_PATTERN.match(entity) and entity in entities:
                next_hops.append(entity)
                entity=entities[entity]                
            relations.append(entities[root]+' '+relation+' '+entity)
    return(relations, next_hops)
                
def rel_hopper(kg, root, entities, max_hops=3):
    cur_hops = root
    relations = []
    visited_hops = set()
    for stage in range(max_hops):
        next_hops = []
        for hop in cur_hops:
            prop_relations, prop_hops = fact_extractor(kg, hop, entities)
            visited_hops.add(hop)
            prop_hops = filter(lambda hop: hop not in visited_hops, prop_hops) 
            relations.extend(prop_relations)
            next_hops.extend(prop_hops)            
        cur_hops = next_hops
    
    return(relations)
    


class RelationExtractor:
    def __init__(self):
        self.kg, denotationals, self.entities = load_kg('.')
        self.denotationals = dict((k.lower(), 
                                   v) for k,v in denotationals.items())
    
    def getRelations(self, names, hops=3):
        root = [self.denotationals[name.lower()] for name in names]
        return(rel_hopper(self.kg, root, self.entities)[:200])
