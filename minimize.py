from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import os
import sys
import json
import tempfile
import subprocess
import collections

import util
import conll
from bert import tokenization

class DocumentState(object):
  def __init__(self, key):
    self.doc_key = key
    self.sentence_end = []
    self.token_end = []
    self.tokens = []
    self.subtokens = []
    self.info = []
    self.segments = []
    self.subtoken_map = []
    self.segment_subtoken_map = []
    self.sentence_map = []
    self.pronouns = []
    self.clusters = collections.defaultdict(list)
    self.coref_stacks = collections.defaultdict(list)
    self.speakers = []
    self.segment_info = []

  def finalize(self):
    # finalized: segments, segment_subtoken_map
    # populate speakers from info
    subtoken_idx = 0
    for segment in self.segment_info:
      speakers = []
      for i, tok_info in enumerate(segment):
        if tok_info is None and (i == 0 or i == len(segment) - 1):
          speakers.append('[SPL]')
        elif tok_info is None:
          speakers.append(speakers[-1])
        else:
          speakers.append(tok_info[9])
          if tok_info[4] == 'PRP':
            self.pronouns.append(subtoken_idx)
        subtoken_idx += 1
      self.speakers += [speakers]
    # populate sentence map

    # populate clusters
    first_subtoken_index = -1
    for seg_idx, segment in enumerate(self.segment_info):
      speakers = []
      for i, tok_info in enumerate(segment):
        first_subtoken_index += 1
        coref = tok_info[-2] if tok_info is not None else '-'
        if coref != "-":
          last_subtoken_index = first_subtoken_index + tok_info[-1] - 1
          for part in coref.split("|"):
            if part[0] == "(":
              if part[-1] == ")":
                cluster_id = int(part[1:-1])
                self.clusters[cluster_id].append((first_subtoken_index, last_subtoken_index))
              else:
                cluster_id = int(part[1:])
                self.coref_stacks[cluster_id].append(first_subtoken_index)
            else:
              cluster_id = int(part[:-1])
              start = self.coref_stacks[cluster_id].pop()
              self.clusters[cluster_id].append((start, last_subtoken_index))
    # merge clusters
    merged_clusters = []
    for c1 in self.clusters.values():
      existing = None
      for m in c1:
        for c2 in merged_clusters:
          if m in c2:
            existing = c2
            break
        if existing is not None:
          break
      if existing is not None:
        print("Merging clusters (shouldn't happen very often.)")
        existing.update(c1)
      else:
        merged_clusters.append(set(c1))
    merged_clusters = [list(c) for c in merged_clusters]
    all_mentions = util.flatten(merged_clusters)
    sentence_map =  get_sentence_map(self.segments, self.sentence_end)
    subtoken_map = util.flatten(self.segment_subtoken_map)
    assert len(all_mentions) == len(set(all_mentions))
    num_words =  len(util.flatten(self.segments))
    assert num_words == len(util.flatten(self.speakers))
    assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
    assert num_words == len(sentence_map), (num_words, len(sentence_map))
    return {
      "doc_key": self.doc_key,
      "sentences": self.segments,
      "speakers": self.speakers,
      "constituents": [],
      "ner": [],
      "clusters": merged_clusters,
      'sentence_map':sentence_map,
      "subtoken_map": subtoken_map,
      'pronouns': self.pronouns
    }


def normalize_word(word, language):
  if language == "arabic":
    word = word[:word.find("#")]
  if word == "/." or word == "/?":
    return word[1:]
  else:
    return word

# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
  current = 0
  previous_token = 0
  while current < len(document_state.subtokens):
    end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
    while end >= current and not constraints1[end]:
      end -= 1
    if end < current:
        end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
        while end >= current and not constraints2[end]:
            end -= 1
        if end < current:
            raise Exception("Can't find valid segment")
    document_state.segments.append(['[CLS]'] + document_state.subtokens[current:end + 1] + ['[SEP]'])
    subtoken_map = document_state.subtoken_map[current : end + 1]
    document_state.segment_subtoken_map.append([previous_token] + subtoken_map + [subtoken_map[-1]])
    info = document_state.info[current : end + 1]
    document_state.segment_info.append([None] + info + [None])
    current = end + 1
    previous_token = subtoken_map[-1]

def get_sentence_map(segments, sentence_end):
  current = 0
  sent_map = []
  sent_end_idx = 0
  assert len(sentence_end) == sum([len(s) -2 for s in segments])
  for segment in segments:
    sent_map.append(current)
    for i in range(len(segment) - 2):
      sent_map.append(current)
      current += int(sentence_end[sent_end_idx])
      sent_end_idx += 1
    sent_map.append(current)
  return sent_map

def get_document(document_lines, tokenizer, language, segment_len):
  document_state = DocumentState(document_lines[0])
  word_idx = -1
  for line in document_lines[1]:
    row = line.split()
    sentence_end = len(row) == 0
    if not sentence_end:
      assert len(row) >= 12
      word_idx += 1
      word = normalize_word(row[3], language)
      subtokens = tokenizer.tokenize(word)
      document_state.tokens.append(word)
      document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
      for sidx, subtoken in enumerate(subtokens):
        document_state.subtokens.append(subtoken)
        info = None if sidx != 0 else (row + [len(subtokens)])
        document_state.info.append(info)
        document_state.sentence_end.append(False)
        document_state.subtoken_map.append(word_idx)
    else:
      document_state.sentence_end[-1] = True
  # split_into_segments(document_state, segment_len, document_state.token_end)
  # split_into_segments(document_state, segment_len, document_state.sentence_end)
  constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
  split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
  stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
  document = document_state.finalize()
  return document

def skip(doc_key):
  # if doc_key in ['nw/xinhua/00/chtb_0078_0', 'wb/eng/00/eng_0004_1']: #, 'nw/xinhua/01/chtb_0194_0', 'nw/xinhua/01/chtb_0157_0']:
    # return True
  return False

def minimize_partition(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir):
  input_path = "{}/{}.{}.{}_{}triggers".format(input_dir, name, language, extension, sys.argv[5])
  output_path = "{}/{}.{}.{}_{}triggers.jsonlines".format(output_dir, name, language, seg_len, sys.argv[5])
  if 'stereotype' in name:
    input_path ="{}/{}.{}_{}triggers".format(input_dir, name, extension, sys.argv[5])
    output_path = "{}/{}.{}_{}triggers.jsonlines".format(output_dir, name,  seg_len, sys.argv[5]) 
  count = 0
  print("Minimizing {}".format(input_path))
  documents = []
  with open(input_path, "r") as input_file:
    for line in input_file.readlines():
      begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
      if begin_document_match:
        doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
        documents.append((doc_key, []))
      elif line.startswith("#end document"):
        continue
      else:
        documents[-1][1].append(line)
  
  with open(output_path, "w") as output_file:
    for document_lines in documents:
      if skip(document_lines[0]):
        continue
      document = get_document(document_lines, tokenizer, language, seg_len)
      output_file.write(json.dumps(document))
      output_file.write("\n")
      count += 1
  print("Wrote {} documents to {}".format(count, output_path))

def minimize_conll_jy(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir, trigger_tokens=None):
  input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
  if 'stereotype' in name:
    input_path ="{}/{}.{}".format(input_dir, name,  extension)
  # output_path = "{}/{}.{}_triggers.jsonlines".format(output_dir, name, seg_len)
  output_path = input_path + "_"+ str(len(trigger_tokens))+"triggers"
  count = 0
  print("adding triggered lines to {}".format(input_path))
  documents = []
  with open(input_path, "r") as input_file:
    for line in input_file.readlines():
      begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
      if begin_document_match:
        doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
        documents.append(line)
        part_id = str(int(begin_document_match.group(2)) % 100)
        if trigger_tokens != None:
          for idx in range(len(trigger_tokens)):
            documents.append('\t'.join([begin_document_match.group(1), part_id, str(-len(trigger_tokens) + idx), trigger_tokens[idx], '-', '-', '-', '-', '-', 'Speaker#1', '*', '*', '*', '*', '-'])+"\n")
      elif line.startswith("#end document"):
        documents.append(line)
      else:
        documents.append(line)
    print("documents:", documents[:5])
  
  with open(output_path, "w") as output_file:
    for document_lines in documents:
      output_file.write(document_lines)
      count += 1
  print("Wrote {} documents to {}".format(count, output_path))


def minimize_language(filename,language, extension, labels, stats, vocab_file, seg_len, input_dir, output_dir, do_lower_case, triggers):
  # do_lower_case = True if 'chinese' in vocab_file else False
  tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)
  # minimize_partition("dev.union", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  # minimize_partition("train.union", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  # minimize_partition("test", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_conll_jy(filename, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir, triggers)
  minimize_partition(filename, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir)

if __name__ == "__main__":
  vocab_file = sys.argv[1]
  input_dir = sys.argv[2]
  output_dir = sys.argv[3]
  do_lower_case = sys.argv[4].lower() == 'true'
  triggers = ["the"]* int(sys.argv[5])
  print(do_lower_case)
  labels = collections.defaultdict(set)
  stats = collections.defaultdict(int)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  # for seg_len in [128, 256, 384, 512]:
  for seg_len in [384]:
    # minimize_language("test", "english", "v4_gold_conll",  labels, stats, vocab_file, \
      # seg_len, input_dir, output_dir, do_lower_case, triggers)
    minimize_language("test_type2_anti_stereotype", "english", "v4_auto_conll",  labels, stats, vocab_file, \
      seg_len, input_dir, output_dir, do_lower_case, triggers)
    # minimize_language("chinese", labels, stats, vocab_file, seg_len)
    # minimize_language("es", labels, stats, vocab_file, seg_len)
    # minimize_language("arabic", labels, stats, vocab_file, seg_len)
  for k, v in labels.items():
    print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
  for k, v in stats.items():
    print("{} = {}".format(k, v))
