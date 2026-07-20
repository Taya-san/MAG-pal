# memory_store.py
# MEMORY ENGINE — SQLite-backed hierarchical memory with LDA classification.
#
# This module stores AI responses as a tree of blocks + sentences.
# It works with multilingual-e5-small embeddings and a pre-trained LDA
# to automatically flag definitional sentences worth remembering.
#
# Why block + sentence?
#   AI responses have structure: a paragraph introduces a concept (block),
#   followed by list items, code blocks, or tables (child blocks).
#   Each individual line/sentence gets its own embedding for fine-grained
#   retrieval, but blocks group them for context.
#
# Schema:
#   sentences: one row per sentence or line
#     Stores: text, embedding (384-dim float32 blob), LDA score, label
#     Label: 'flagged' (worth storing) or 'unlabeled' (skip)
#   blocks: groups of related sentences
#     Types: paragraph, list, code, table, equation
#     parent_id: links child blocks to their parent
#   block_relations: parent-child links
#
# Pipeline:
#   AI response text
#     -> BlockParser.parse_and_store()
#     -> Split into blocks (code/list/table/paragraph/equation)
#     -> Split each block into individual sentences
#     -> Embed each sentence with multilingual-e5-small
#     -> Classify with LDA (pre-trained on 142 examples)
#     -> Code/equations/table rows inherit from parent block
#
# Detection order (first match wins):
#   1. Code blocks (``` fences)
#   2. List items (numbered or bulleted)
#   3. Tables (pipe-delimited | rows)
#   4. Paragraph parents (lines ending with ':')
#   5. Equations (contain '=' but not ends with '.')
#   6. Regular sentences (everything else)
#
# Cross-lingual: LDA trained on English but transfers to Indonesian
# because multilingual-e5-small maps both languages into the same
# embedding space. Indonesian stopwords included.
"""
Memory Store — SQLite-backed hierarchical memory for the MAG-pal bot.

Schema:
  sentences: one row per sentence or line of text
    id | text | block_id | type | embedding | score | label | created_at

  blocks: groups of sentences forming a logical unit
    id | text | type | parent_id | top_words | sentiment | created_at

  block_relations: parent-child links between blocks
    id | parent_id | child_id | relation_type

Each sentence gets its own embedding and LDA score.
Blocks group related sentences and form parent-child hierarchies.
"""

import sqlite3, json, re, time, numpy as np, pickle
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DB_PATH = "/home/taya/Projects/MAG-pal/memory.db"

# ===== STOPWORDS =====
STOPS = ['a', 'about', 'above', 'across', 'ada', 'adalah', 'adanya', 'adapun', 'after', 'afterwards', 'again', 'against', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya', 'aku', 'akulah', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'amat', 'amatlah', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'anda', 'andalah', 'another', 'antar', 'antara', 'antaranya', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'are', 'around', 'artinya', 'as', 'asal', 'asalkan', 'at', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'back', 'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya', 'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah', 'be', 'beberapa', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'begini', 'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun', 'behind', 'being', 'bekerja', 'belakang', 'belakangan', 'below', 'belum', 'belumlah', 'benar', 'benarkah', 'benarlah', 'berada', 'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah', 'berapalah', 'berapapun', 'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya', 'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan', 'berlalu', 'berlangsung', 'berlebihan', 'bermacam', 'bermacam-macam', 'bermaksud', 'bermula', 'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 'berturut', 'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'beside', 'besides', 'betul', 'betulkah', 'between', 'beyond', 'biasa', 'biasanya', 'bila', 'bilakah', 'bill', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah', 'both', 'bottom', 'buat', 'bukan', 'bukankah', 'bukanlah', 'bukannya', 'bulan', 'bung', 'but', 'by', 'call', 'came', 'can', 'cannot', 'cant', 'cara', 'caranya', 'co', 'come', 'con', 'could', 'couldnt', 'cry', 'cukup', 'cukupkah', 'cukuplah', 'cuma', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang', 'de', 'dekat', 'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'describe', 'detail', 'di', 'dia', 'diakhiri', 'diakhirinya', 'dialah', 'diantara', 'diantaranya', 'diberi', 'diberikan', 'diberikannya', 'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'digunakan', 'diibaratkan', 'diibaratkannya', 'diingat', 'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya', 'dikarenakan', 'dikatakan', 'dikatakannya', 'dikerjakan', 'diketahui', 'diketahuinya', 'dikira', 'dilakukan', 'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan', 'dimaksudkannya', 'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan', 'dimulai', 'dimulailah', 'dimulainya', 'dimungkinkan', 'dini', 'dipastikan', 'diperbuat', 'diperbuatnya', 'dipergunakan', 'diperkirakan', 'diperlihatkan', 'diperlukan', 'diperlukannya', 'dipersoalkan', 'dipertanyakan', 'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan', 'disebutkannya', 'disini', 'disinilah', 'ditambahkan', 'ditandaskan', 'ditanya', 'ditanyai', 'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjuk', 'ditunjuki', 'ditunjukkan', 'ditunjukkannya', 'ditunjuknya', 'dituturkan', 'dituturkannya', 'diucapkan', 'diucapkannya', 'diungkapkan', 'do', 'done', 'dong', 'down', 'dua', 'due', 'dulu', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empat', 'empty', 'enggak', 'enggaknya', 'enough', 'entah', 'entahlah', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'explain', 'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'gone', 'got', 'guna', 'gunakan', 'had', 'hal', 'hampir', 'hanya', 'hanyalah', 'hari', 'harus', 'haruslah', 'harusnya', 'has', 'hasnt', 'have', 'he', 'hence', 'hendak', 'hendaklah', 'hendaknya', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'hingga', 'his', 'how', 'however', 'hundred', 'i', 'ia', 'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'ibu', 'ie', 'if', 'ikut', 'in', 'inc', 'indeed', 'ingat', 'ingat-ingat', 'ingin', 'inginkah', 'inginkan', 'ini', 'inikah', 'inilah', 'interest', 'into', 'is', 'it', 'its', 'itself', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah', 'jadinya', 'jangan', 'jangankan', 'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas', 'jelaskan', 'jelaslah', 'jelasnya', 'jika', 'jikalau', 'juga', 'jumlah', 'jumlahnya', 'justru', 'kala', 'kalau', 'kalaulah', 'kalaupun', 'kalian', 'kami', 'kamilah', 'kamu', 'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun', 'karena', 'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah', 'katanya', 'ke', 'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keep', 'keinginan', 'kelamaan', 'kelihatan', 'kelihatannya', 'kelima', 'keluar', 'kembali', 'kemudian', 'kemungkinan', 'kemungkinannya', 'kenapa', 'kepada', 'kepadanya', 'kesampaian', 'keseluruhan', 'keseluruhannya', 'keterlaluan', 'ketika', 'khususnya', 'kini', 'kinilah', 'kira', 'kira-kira', 'kiranya', 'kita', 'kitalah', 'know', 'kok', 'kurang', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu', 'lama', 'lamanya', 'lanjut', 'lanjutnya', 'last', 'latter', 'latterly', 'least', 'lebih', 'less', 'let', 'lewat', 'like', 'lima', 'ltd', 'luar', 'macam', 'made', 'maka', 'makanya', 'make', 'makin', 'making', 'malah', 'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'many', 'masa', 'masalah', 'masalahnya', 'masih', 'masihkah', 'masing', 'masing-masing', 'mau', 'maupun', 'may', 'me', 'meanwhile', 'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya', 'memang', 'memastikan', 'memberi', 'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta', 'memintakan', 'memisalkan', 'memperbuat', 'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan', 'mempersoalkan', 'mempertanyakan', 'mempunyai', 'memulai', 'memungkinkan', 'menaiki', 'menambahkan', 'menandaskan', 'menanti', 'menanti-nanti', 'menantikan', 'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan', 'mendatang', 'mendatangi', 'mendatangkan', 'menegaskan', 'mengakhiri', 'mengapa', 'mengatakan', 'mengatakannya', 'mengenai', 'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki', 'mengibaratkan', 'mengibaratkannya', 'mengingat', 'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan', 'mengucapkannya', 'mengungkapkan', 'menjadi', 'menjawab', 'menjelaskan', 'menuju', 'menunjuk', 'menunjuki', 'menunjukkan', 'menunjuknya', 'menurut', 'menuturkan', 'menyampaikan', 'menyangkut', 'menyatakan', 'menyebutkan', 'menyeluruh', 'menyiapkan', 'merasa', 'mereka', 'merekalah', 'merupakan', 'meski', 'meskipun', 'meyakini', 'meyakinkan', 'might', 'mill', 'mine', 'minta', 'mirip', 'misal', 'misalkan', 'misalnya', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'mula', 'mulai', 'mulailah', 'mulanya', 'mungkin', 'mungkinkah', 'must', 'my', 'myself', 'nah', 'naik', 'name', 'namely', 'namun', 'nanti', 'nantinya', 'need', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'nyaris', 'nyatanya', 'of', 'off', 'often', 'oleh', 'olehnya', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'pada', 'padahal', 'padanya', 'pak', 'paling', 'panjang', 'pantas', 'para', 'part', 'pasti', 'pastilah', 'penting', 'pentingnya', 'per', 'percuma', 'perhaps', 'perlu', 'perlukah', 'perlunya', 'pernah', 'persoalan', 'pertama', 'pertama-tama', 'pertanyaan', 'pertanyakan', 'pihak', 'pihaknya', 'please', 'pukul', 'pula', 'pun', 'punya', 'put', 'rasa', 'rasanya', 'rata', 'rather', 're', 'rupanya', 'saat', 'saatnya', 'said', 'saja', 'sajalah', 'saling', 'sama', 'sama-sama', 'sambil', 'same', 'sampai', 'sampai-sampai', 'sampaikan', 'sana', 'sangat', 'sangatlah', 'satu', 'say', 'saya', 'sayalah', 'se', 'sebab', 'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik', 'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum', 'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar', 'sebetulnya', 'sebisanya', 'sebuah', 'sebut', 'sebutlah', 'sebutnya', 'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian', 'sedikit', 'sedikitnya', 'see', 'seem', 'seemed', 'seeming', 'seems', 'seenaknya', 'segala', 'segalanya', 'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh', 'sejenak', 'sejumlah', 'sekadar', 'sekadarnya', 'sekali', 'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang', 'sekecil', 'seketika', 'sekiranya', 'sekitar', 'sekitarnya', 'sekurang-kurangnya', 'sekurangnya', 'sela', 'selagi', 'selain', 'selaku', 'selalu', 'selama', 'selama-lamanya', 'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya', 'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya', 'sementara', 'semisal', 'semisalnya', 'sempat', 'semua', 'semuanya', 'semula', 'sendiri', 'sendirian', 'sendirinya', 'seolah', 'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti', 'sepertinya', 'sepihak', 'sering', 'seringnya', 'serious', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera', 'sesekali', 'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah', 'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi', 'seusai', 'several', 'sewaktu', 'shall', 'she', 'should', 'show', 'siap', 'siapa', 'siapakah', 'siapapun', 'side', 'since', 'sincere', 'sini', 'sinilah', 'six', 'sixty', 'so', 'soal', 'soalnya', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'suatu', 'such', 'sudah', 'sudahkah', 'sudahlah', 'supaya', 'system', 'tadi', 'tadinya', 'tahu', 'tahun', 'tak', 'take', 'taken', 'tambah', 'tambahnya', 'tampak', 'tampaknya', 'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya', 'tapi', 'tegas', 'tegasnya', 'telah', 'tempat', 'ten', 'tengah', 'tentang', 'tentu', 'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa', 'terbanyak', 'terdahulu', 'terdapat', 'terdiri', 'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah', 'terjadinya', 'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan', 'tersebut', 'tersebutlah', 'tertentu', 'tertuju', 'terus', 'terutama', 'tetap', 'tetapi', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'thing', 'things', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'tiap', 'tiba', 'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah', 'tiga', 'tinggi', 'to', 'together', 'toh', 'too', 'took', 'top', 'toward', 'towards', 'tunjuk', 'turut', 'tutur', 'tuturnya', 'twelve', 'twenty', 'two', 'ucap', 'ucapnya', 'ujar', 'ujarnya', 'umum', 'umumnya', 'un', 'under', 'ungkap', 'ungkapnya', 'until', 'untuk', 'up', 'upon', 'us', 'usah', 'usai', 'use', 'used', 'using', 'very', 'via', 'waduh', 'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun', 'was', 'way', 'we', 'well', 'went', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'wong', 'would', 'yaitu', 'yakin', 'yakni', 'yang', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves']


class MemoryStore:
    """SQLite-backed store for sentences, blocks, and their relationships."""

    def __init__(self, db_path=DB_PATH, lda_path=None):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self.analyzer = SentimentIntensityAnalyzer()
        self.lda = None
        if lda_path:
            self.load_lda(lda_path)

    def _create_tables(self):
        # Create the schema if it doesn't exist.
        # Uses IF NOT EXISTS so it's safe to call every startup.
        # SQLite auto-creates the file if it doesn't exist.
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'paragraph',
                parent_id INTEGER REFERENCES blocks(id),
                top_words TEXT DEFAULT '[]',
                sentiment REAL DEFAULT 0.0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                block_id INTEGER REFERENCES blocks(id),
                line_number INTEGER DEFAULT 0,
                type TEXT NOT NULL DEFAULT 'sentence',
                embedding BLOB,
                score REAL DEFAULT 0.0,
                label TEXT DEFAULT 'unlabeled',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS block_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER REFERENCES blocks(id),
                child_id INTEGER REFERENCES blocks(id),
                relation_type TEXT DEFAULT 'child_of'
            );

            CREATE INDEX IF NOT EXISTS idx_sentences_block ON sentences(block_id);
            CREATE INDEX IF NOT EXISTS idx_blocks_parent ON blocks(parent_id);
            CREATE INDEX IF NOT EXISTS idx_relations_parent ON block_relations(parent_id);
            CREATE INDEX IF NOT EXISTS idx_relations_child ON block_relations(child_id);
        """)
        self.conn.commit()

    def extract_top_words(self, text, max_words=5, content_type='paragraph'):
        # Extract top words from text, excluding stopwords.
        # For tables: prioritizes column headers and first-column values
        # because table content rarely has repeated words (frequency fails).
        # For everything else: standard frequency-based extraction.
        if content_type == 'table':
            return self._extract_table_top_words(text, max_words)
        
        cleaned = re.sub(r'[^a-z\s]', ' ', text.lower())
        words = [w for w in cleaned.split() if w not in STOPS and len(w) > 2]
        if not words:
            return []
        return [w for w, _ in Counter(words).most_common(max_words)]

    def _extract_table_top_words(self, text, max_words=5):
        # Tables don't have repeated words, so frequency doesn't work.
        # Extract column headers + non-generic first-column values.
        # Skip: numbers, generic labels (positive, negative, yes, no, high, low).
        # Keep: named entities (LU, SVD, SGD), property names (learning rate).
        rows = text.strip().split('\n')
        candidates = []
        
        # Generic table labels that don't describe content
        generic_labels = {'positive', 'negative', 'neutral', 'high', 'low', 'medium',
                          'yes', 'no', 'true', 'false', 'good', 'bad', 'best', 'worst',
                          'fast', 'slow', 'small', 'large', 'big', 'short', 'long',
                          'type', 'value', 'name', 'desc', 'description'}
        
        for i, row in enumerate(rows):
            cells = [c.strip().lower() for c in row.split('|') if c.strip()]
            if not cells:
                continue
            # First row = column headers (most important, always include)
            if i == 0:
                candidates.extend(cells)
            # First column values = entities being compared
            if cells:
                val = cells[0]
                # Skip separator rows (|---|)
                if val.replace('-', '').strip() == '':
                    pass
                # Skip numeric values (1, 2, 3 or O(n^3) etc.)
                elif re.match(r'^[\d()\^ons\s]+$', val):
                    pass  # skip
                # Skip generic labels
                elif val.strip() not in generic_labels:
                    candidates.append(val)
            # Skip separator rows (|---|---|) — any number of dashes
            if all(c.strip().replace('-', '') == '' for c in cells):
                continue
        
        all_words = []
        for c in candidates:
            words = re.sub(r'[^a-z\s]', ' ', c).split()
            all_words.extend([w for w in words if w not in STOPS and len(w) > 2])
        
        seen = set()
        result = []
        for w in all_words:
            if w not in seen:
                seen.add(w)
                result.append(w)
        
        return result[:max_words]

    # ===== INSERT =====

    def add_block(self, text, block_type='paragraph', parent_id=None):
        # Insert a block and return its ID.
        # A block is a group of related lines: paragraph, code, list, table, equation.
        # parent_id links this block as a child of another block.
        top_words = json.dumps(self.extract_top_words(text, content_type=block_type))
        sentiment = self.analyzer.polarity_scores(text)['compound']
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO blocks (text, type, parent_id, top_words, sentiment, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (text.strip(), block_type, parent_id, top_words, sentiment, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def add_sentence(self, text, block_id, line_number=0, sent_type='sentence',
                     embedding=None, score=0.0, label='unlabeled', strip=True):
        # Insert a single sentence or line into the sentences table.
        # Each sentence gets its own embedding (384-dim float32 blob).
        # strip=False preserves indentation for code lines.
        # label is set later by LDA classification.
        text_content = text.strip() if strip else text
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO sentences (text, block_id, line_number, type, embedding, score, label, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (text_content, block_id, line_number, sent_type, emb_blob, float(score), label, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def add_relation(self, parent_id, child_id, relation_type='child_of'):
        # Link two blocks as parent-child.
        # relation_type = 'child_of' is the default.
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO block_relations (parent_id, child_id, relation_type) VALUES (?, ?, ?)",
            (parent_id, child_id, relation_type)
        )
        self.conn.commit()
        return cur.lastrowid

    # ===== QUERY =====

    def get_block(self, block_id):
        """Get a block by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM blocks WHERE id = ?", (block_id,))
        return cur.fetchone()

    def get_block_children(self, block_id):
        """Get all child blocks of a block."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT b.* FROM blocks b
            JOIN block_relations r ON b.id = r.child_id
            WHERE r.parent_id = ? AND r.relation_type = 'child_of'
            ORDER BY b.id
        """, (block_id,))
        return cur.fetchall()

    def get_sentences_by_block(self, block_id):
        """Get all sentences in a block, ordered by line_number."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM sentences WHERE block_id = ? ORDER BY line_number",
            (block_id,)
        )
        return cur.fetchall()

    def get_labeled(self, label='flagged', limit=50):
        """Get the most recent sentences with a given label."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT s.*, b.type as block_type, b.top_words
            FROM sentences s
            JOIN blocks b ON s.block_id = b.id
            WHERE s.label = ?
            ORDER BY s.created_at DESC
            LIMIT ?
        """, (label, limit))
        return cur.fetchall()

    def search_by_keyword(self, keyword, label='flagged', limit=20):
        """Find labeled sentences containing a keyword."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT s.*, b.type as block_type
            FROM sentences s
            JOIN blocks b ON s.block_id = b.id
            WHERE s.label = ? AND s.text LIKE ?
            ORDER BY s.score DESC
            LIMIT ?
        """, (label, f'%{keyword}%', limit))
        return cur.fetchall()

    def get_tree(self, block_id=None):
        """Get the full block tree (root if None)."""
        cur = self.conn.cursor()
        if block_id is None:
            cur.execute("SELECT * FROM blocks WHERE parent_id IS NULL ORDER BY id")
        else:
            cur.execute("SELECT * FROM blocks WHERE id = ?", (block_id,))
        roots = cur.fetchall()
        
        result = []
        for root in roots:
            node = dict(root)
            node['children'] = self.get_block_children(root['id'])
            result.append(node)
        return result

    def load_lda(self, lda_path):
        """Load a pre-trained LDA model from file."""
        with open(lda_path, 'rb') as f:
            self.lda = pickle.load(f)

    def count_stats(self):
        """Return counts of stored data."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sentences")
        s = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM blocks")
        b = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM sentences WHERE label != 'unlabeled'")
        fl = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM block_relations")
        r = cur.fetchone()[0]
        return {'sentences': s, 'blocks': b, 'flagged': fl, 'relations': r}

    def close(self):
        self.conn.close()


# ===== BLOCK PARSER (integrated with MemoryStore) =====

class BlockParser:
    # Parses AI response text and stores directly into MemoryStore.
    #
    # Detection order (first match wins):
    #   1. Code blocks (``` ... ```) — multi-line, becomes a child block
    #   2. List items (1., -, *) — each item is a sentence in a list block
    #   3. Tables (| ... |) — multi-line, becomes a child block
    #   4. Paragraph parents (lines ending with ':') — introduces children
    #   5. Equations (contain '=') — single line, becomes a child block
    #   6. Regular sentences — flat paragraphs or continuation text
    #
    # After parsing, each sentence is classified by LDA.
    # Code lines, equations, and table rows SKIP LDA and inherit
    # their label from the parent block's first natural-language sentence.
    #
    # This is because the embedding model doesn't understand
    # programming syntax or mathematical notation — it would get
    # meaningless scores.

    def __init__(self, memory_store, embed_fn=None):
        # store: MemoryStore instance where data will be inserted.
        # embed_fn: function(text) -> 384-dim numpy array.
        self.store = memory_store
        self.embed = embed_fn

    def parse_and_store(self, text):
        """Parse text and store everything into the database.
        
        Returns: (block_ids, sentence_ids) for the created records.
        """
        lines = text.split('\n')
        all_block_ids = []
        all_sentence_ids = []
        i = 0
        current_block_id = None  # block collecting children
        
        while i < len(lines):
            stripped = lines[i].strip()
            
            # --- Empty line: finalize current block ---
            if not stripped:
                current_block_id = None
                i += 1
                continue
            
            # --- Code block ---
            if stripped.startswith('```'):
                code_lines = [lines[i]]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_lines.append(lines[i])
                    i += 1
                
                code_text = '\n'.join(code_lines)
                block_id = self.store.add_block(code_text, 'code', current_block_id)
                all_block_ids.append(block_id)
                
                # Each line of code becomes a sentence
                for ln, code_line in enumerate(code_lines):
                    sid = self.store.add_sentence(
                        code_line, block_id, ln, 'code_line',
                        score=0.0, label='unlabeled', strip=False
                    )
                    all_sentence_ids.append(sid)
                
                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                current_block_id = None
                continue
            
            # --- List item ---
            if re.match(r'^\s*(?:\d+[\.\)]|[-*])\s', stripped):
                if current_block_id is None:
                    # Create a parent block for orphan list items
                    # Store first item text so top_words are meaningful
                    block_id = self.store.add_block(stripped, 'list', None)
                    current_block_id = block_id
                    all_block_ids.append(block_id)
                    if current_block_id:
                        pass  # already set
                
                line_num = len(self.store.get_sentences_by_block(current_block_id))
                sid = self.store.add_sentence(
                    stripped, current_block_id, line_num, 'list_item',
                    score=0.0, label='unlabeled'
                )
                all_sentence_ids.append(sid)
                # Update block text to include this list item (preserves top_words)
                cur = self.store.conn.cursor()
                cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
                existing = cur.fetchone()[0]
                new_text = existing + ' ' + stripped if existing else stripped
                cur.execute("UPDATE blocks SET text = ?, top_words = ? WHERE id = ?",
                    (new_text, json.dumps(self.store.extract_top_words(new_text)), current_block_id))
                self.store.conn.commit()
                i += 1
                continue
            
            # --- Table ---
            if stripped.startswith('|') and stripped.endswith('|'):
                table_lines = [lines[i]]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                    table_lines.append(lines[i])
                    i += 1
                
                table_text = '\n'.join(table_lines)
                block_id = self.store.add_block(table_text, 'table', current_block_id)
                all_block_ids.append(block_id)
                
                for ln, tl in enumerate(table_lines):
                    sid = self.store.add_sentence(tl, block_id, ln, 'table_row')
                    all_sentence_ids.append(sid)
                
                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                current_block_id = None
                continue
            
            # --- Regular sentence (continuation or new paragraph) ---
            if current_block_id is not None and not stripped.endswith(':'):
                # Continuation: add each sentence individually to current block
                for ln, s in enumerate(re.split(r'(?<=[.!?])\s+(?=[A-Z"(\[])', stripped)):
                    s = s.strip()
                    if not s:
                        continue
                    sid = self.store.add_sentence(s, current_block_id, ln, 'sentence')
                    all_sentence_ids.append(sid)
                # Update block text (full merged paragraph)
                cur = self.store.conn.cursor()
                cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
                existing = cur.fetchone()[0]
                new_text = existing + ' ' + stripped
                cur.execute("UPDATE blocks SET text = ?, top_words = ?, sentiment = ? WHERE id = ?",
                    (new_text, json.dumps(self.store.extract_top_words(new_text)),
                     self.store.analyzer.polarity_scores(new_text)['compound'], current_block_id))
                self.store.conn.commit()
            else:
                # New paragraph: split into individual sentences
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"(\[])', stripped) if len(s.strip()) > 3]
                block_text = ' '.join(sents)
                block_id = self.store.add_block(block_text, 'paragraph', None)
                all_block_ids.append(block_id)
                for ln, s in enumerate(sents):
                    sid = self.store.add_sentence(s, block_id, ln, 'sentence')
                    all_sentence_ids.append(sid)
                current_block_id = block_id
            
            i += 1
        
        # Now classify each sentence with LDA if available
        if self.embed and self.store.lda:
            self._classify_sentences(all_sentence_ids)
        
        return all_block_ids, all_sentence_ids

    def _classify_sentences(self, sentence_ids):
        """Run LDA on all stored sentences and set their label.
        
        Skips: code lines, equations, table rows, code fences — these
        contain non-natural language that the embedding model can't
        understand. They inherit their parent block's classification instead.
        
        Currently labels as 'flagged' if LDA score > 0.
        """
        cur = self.store.conn.cursor()
        for sid in sentence_ids:
            cur.execute("SELECT s.text, s.type, b.id as block_id FROM sentences s JOIN blocks b ON s.block_id = b.id WHERE s.id = ?", (sid,))
            row = cur.fetchone()
            if not row:
                continue
            text, stype, block_id = row[0], row[1], row[2]
            
            # Skip non-natural-language types: code lines, equations, table rows, fences
            # These contain programming syntax or math symbols that the embedding
            # model doesn't understand. Inherit classification from parent block.
            if stype in ('code_line', 'equation', 'table_row'):
                # Find parent block of this block via block_relations
                cur.execute("SELECT parent_id FROM block_relations WHERE child_id = ? AND relation_type = 'child_of'", (block_id,))
                parent_rel = cur.fetchone()
                if parent_rel:
                    parent_id = parent_rel[0]
                    # Get first natural-language sentence from parent
                    cur.execute("SELECT id, score, label FROM sentences WHERE block_id = ? AND type = 'sentence' ORDER BY line_number LIMIT 1", (parent_id,))
                    ps = cur.fetchone()
                    if ps:
                        cur.execute("UPDATE sentences SET score = ?, label = ? WHERE id = ?", (ps[1], ps[2], sid))
                        continue
                # Fallback: mark as unlabeled with score 0
                cur.execute("UPDATE sentences SET score = 0.0, label = 'unlabeled' WHERE id = ?", (sid,))
                continue
            
            if len(text.strip()) < 10:
                continue
            emb = self.embed(text).reshape(1, -1)
            score = float(self.store.lda.decision_function(emb)[0])
            label = 'flagged' if score > 0 else 'unlabeled'
            cur.execute(
                "UPDATE sentences SET score = ?, label = ? WHERE id = ?",
                (score, label, sid)
            )
        self.store.conn.commit()


if __name__ == "__main__":
    print("Testing MemoryStore + BlockParser...")
    store = MemoryStore(":memory:")
    
    parser = BlockParser(store)
    
    test = """A symmetric matrix equals its own transpose. Eigenvalues are always real for symmetric matrices.

Key properties of symmetric matrices:
1. A = A^T is the defining property
2. All eigenvalues are real

The spectral theorem guarantees:
A = Q * Lambda * Q^T

Implementation:
```python
def is_symmetric(A):
    return np.allclose(A, A.T)
```"""
    
    block_ids, sent_ids = parser.parse_and_store(test)
    stats = store.count_stats()
    print(f"\nStored: {stats['sentences']} sentences, {stats['blocks']} blocks, {stats['relations']} relations")
    
    print("\nBlocks tree:")
    for b in store.get_tree():
        print(f"\n  BLOCK {b['id']}: [{b['type']}] \"{b['text'][:60]}...\"")
        print(f"    top_words: {b['top_words']}")
        for c in b['children']:
            print(f"    ├── CHILD {c['id']}: [{c['type']}] \"{c['text'][:50]}...\"")
            print(f"    |   top_words: {c['top_words']}")
        for s in store.get_sentences_by_block(b['id']):
            print(f"    ├── SENT {s['id']}: [{s['type']}] \"{s['text'][:50]}...\"")
    
    store.close()
    print("\nDone!")
