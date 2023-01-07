import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class Example():
    """ Object for describing an example.
    E.g., if `obj = Example(ex: dict)`, where
    
    `ex = {'utt_id': 1, 'manual_transcript': '我要去新光北路', 'asr_1best': '我要去新光北路', 'semantic': [['inform', '终点名称', '新光北路']]},`

    then
    + `obj.utt = '我要去新光北路'` (`ex['asr_1best']`),
    + `obj.slot = {'inform-终点名称': '新光北路'}`, 
    + `obj.tags = ['O', 'O', 'O', 'B-inform-终点名称', 'I-inform-终点名称', 'I-inform-终点名称', 'I-inform-终点名称']`,
    + `obj.slotvalue = ['inform-终点名称-新光北路']`,
    + `obj.input_idx = [21, 40, 41, 60, 229, 151, 83]` (indices of words),
    + `obj.tag_id = [1, 1, 1, 14, 15, 15, 15]` (indices of tags of words).

    Also, the class method `load_dataset` returns a list of `Example` objects from dataset, and the class method 
    `configuration` initializes instances of `Evaluator`, `Vocab`, `Word2vecUtils` and `LabelVocab` as class attributes.
    """    

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        """ Initializes instances of `Evaluator`, `Vocab`, `Word2vecUtils` and `LabelVocab` as class attributes.
        """        
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        """ Returns a list of `Example` objects from dataset.
        """

        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best']
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
