import json

dataset = []

dataset.append({
    "id": 1,
    "name": "paragraphs_merging",
    "text": (
        "The transformer architecture revolutionized natural language processing.\n"
        "\n"
        "Below is a summary of key model sizes:\n"
        "| Model   | Parameters |\n"
        "|---------|------------|\n"
        "| BERT    | 110M       |\n"
        "| GPT-2   | 1.5B       |\n"
        "| GPT-3   | 175B       |\n"
        "| PaLM    | 540B       |"
    )
})

dataset.append({
    "id": 2,
    "name": "paragraph_table_new_paragraph",
    "text": (
        "The results show a clear improvement in accuracy across all benchmarks:\n"
        "| Dataset   | Before | After |\n"
        "|-----------|--------|-------|\n"
        "| MNLI      | 72.3   | 78.1  |\n"
        "| QNLI      | 88.0   | 91.4  |\n"
        "| RTE       | 65.2   | 70.5  |\n"
        "\n"
        "We attribute this improvement to the new attention mechanism."
    )
})

dataset.append({
    "id": 3,
    "name": "lists_as_child_sentences",
    "text": (
        "Key properties of the algorithm:\n"
        "1. It runs in O(n log n) time\n"
        "2. It requires only O(n) auxiliary space\n"
        "3. It is stable for equal keys\n"
        "4. It parallelizes well on GPU hardware"
    )
})

dataset.append({
    "id": 4,
    "name": "code_block_after_colon",
    "text": (
        "Here is an example implementation of the quicksort algorithm:\n"
        "\n"
        "```python\n"
        "def quicksort(arr):\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n"
        "    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n"
        "```"
    )
})

dataset.append({
    "id": 5,
    "name": "equations_single_line",
    "text": (
        "E = mc^2\n"
        "F = ma\n"
        "PV = nRT\n"
        "The above equations are fundamental to physics."
    )
})

dataset.append({
    "id": 6,
    "name": "classifier_columns_true_false",
    "text": (
        "Model evaluation results:\n"
        "| Test Case | Passed |\n"
        "|-----------|--------|\n"
        "| TC-001    | True   |\n"
        "| TC-002    | False  |\n"
        "| TC-003    | True   |\n"
        "| TC-004    | True   |\n"
        "| TC-005    | False  |\n"
        "| TC-006    | True   |\n"
        "| TC-007    | False  |\n"
        "| TC-008    | True   |\n"
        "| TC-009    | True   |\n"
        "| TC-010    | False  |\n"
        "| TC-011    | False  |\n"
        "| TC-012    | True   |"
    )
})

dataset.append({
    "id": 7,
    "name": "numeric_columns_varied",
    "text": (
        "Experimental measurements:\n"
        "| Sample | Concentration | Wavelength | Error |\n"
        "|--------|---------------|------------|-------|\n"
        "| A-01   | 1.23e-6       | 532        | 2.1%  |\n"
        "| A-02   | 4.56e-7       | 633        | 1.8%  |\n"
        "| A-03   | 7.89e-9       | 780        | 3.2%  |\n"
        "| A-04   | 3.21e-10      | 405        | 2.5%  |\n"
        "| A-05   | 9.87e-6       | 980        | 0.9%  |\n"
        "| A-06   | 6.54e-8       | 450        | 4.1%  |\n"
        "| A-07   | 2.34e-11      | 1064       | 1.2%  |"
    )
})

dataset.append({
    "id": 8,
    "name": "entity_columns_few_unique",
    "text": (
        "Language benchmark scores:\n"
        "| Language | Score |\n"
        "|----------|-------|\n"
        "| Python   | 92.3  |\n"
        "| Rust     | 95.1  |\n"
        "| Python   | 88.7  |\n"
        "| Rust     | 93.4  |\n"
        "| Python   | 91.0  |\n"
        "| Rust     | 96.2  |\n"
        "| Python   | 89.5  |\n"
        "| Rust     | 94.8  |\n"
        "| Python   | 90.3  |\n"
        "| Rust     | 97.0  |"
    )
})

dataset.append({
    "id": 9,
    "name": "mixed_content_all_types",
    "text": (
        "We propose a novel approach to few-shot learning.\n"
        "\n"
        "The method has several advantages:\n"
        "1. Requires only 5 examples per class\n"
        "2. No fine-tuning needed\n"
        "3. Works on mobile devices\n"
        "\n"
        "Benchmark results:\n"
        "| Dataset   | Accuracy | F1 Score |\n"
        "|-----------|----------|----------|\n"
        "| Omniglot  | 98.2     | 97.8     |\n"
        "| MiniImage | 92.5     | 91.6     |\n"
        "\n"
        "Below is the core inference function:\n"
        "\n"
        "```python\n"
        "def predict(model, support, query):\n"
        "    emb = model.encode(support + query)\n"
        "    s, q = emb[:len(support)], emb[len(support):]\n"
        "    logits = cosine_similarity(q, s)\n"
        "    return logits.argmax(dim=-1)\n"
        "```\n"
        "\n"
        "The prototype loss is: L = -log(softmax(d(x, c_y)))\n"
        "Where c_y is the class prototype."
    )
})

dataset.append({
    "id": 10,
    "name": "trailing_flat_paragraph",
    "text": (
        "Configuration parameters for the simulation:\n"
        "| Parameter    | Value  |\n"
        "|--------------|--------|\n"
        "| batch_size   | 32     |\n"
        "| learning_rate| 0.001  |\n"
        "| epochs       | 100    |\n"
        "\n"
        "The simulation will terminate after convergence."
    )
})

dataset.append({
    "id": 11,
    "name": "no_colon_parent_flat",
    "text": (
        "The model converged after 50 epochs with a final loss of 0.023.\n"
        "Validation accuracy plateaued at 94.2 percent.\n"
        "We used early stopping with a patience of 5 epochs."
    )
})

dataset.append({
    "id": 12,
    "name": "multi_level_parent_table_list",
    "text": (
        "Below is a comparison of activation functions:\n"
        "| Activation | Range       | Derivative |\n"
        "|------------|-------------|------------|\n"
        "| ReLU       | [0, inf)    | binary     |\n"
        "| Sigmoid    | (0, 1)      | bell       |\n"
        "\n"
        "Their recommended use cases:\n"
        "1. ReLU for hidden layers of deep networks\n"
        "2. Sigmoid for binary classification output\n"
        "3. Tanh for RNN cells"
    )
})

dataset.append({
    "id": 13,
    "name": "indonesian_mixed_language",
    "text": (
        "Hasil eksperimen menunjukkan peningkatan signifikan:\n"
        "| Metode    | Akurasi | F1-Score |\n"
        "|-----------|---------|----------|\n"
        "| BERT-base | 89.2    | 88.7     |\n"
        "| IndoBERT  | 92.1    | 91.5     |\n"
        "| BiLSTM    | 84.6    | 83.9     |\n"
        "| CNN       | 81.3    | 80.8     |\n"
        "\n"
        "Model ini dilatih menggunakan dataset NusaX yang mencakup 12 bahasa daerah.\n"
        "Kami menemukan bahwa transfer learning dari Multilingual BERT memberikan hasil terbaik."
    )
})

dataset.append({
    "id": 14,
    "name": "long_ai_reasoning",
    "text": (
        "Based on the analysis of the provided code, there are several issues that need to be addressed.\n"
        "\n"
        "First, the attention mask is not being applied correctly in the forward pass. The current implementation passes the mask directly to the scaled dot-product attention function, but it should be expanded to match the shape of the attention scores. Specifically, the mask has shape (batch_size, seq_len) but needs to be reshaped to (batch_size, 1, 1, seq_len) to broadcast correctly over the attention heads. This is a common source of bugs in Transformer implementations, and fixing it will likely resolve the NaN gradient issue reported in the training logs.\n"
        "\n"
        "Second, the layer normalization is applied before residual connections rather than after. According to the original Transformer paper and most modern implementations, pre-norm (LayerNorm before the sublayer) is actually preferred for training stability. However, the code uses post-norm which can lead to gradient explosion in deep models. I recommend switching to pre-norm and adding an additional learnable scale parameter.\n"
        "\n"
        "Third, the positional encoding uses a fixed sinusoidal pattern, which is fine for lengths seen during training, but the model is being evaluated on sequences longer than the maximum training length. This causes a sharp drop in performance. The solution is to either use Rotary Position Embeddings (RoPE) or ALiBi, both of which extrapolate better to longer sequences without requiring changes to the core architecture.\n"
        "\n"
        "Let me verify these findings against the reported metrics. The training loss curve shows a sudden spike around epoch 12, which coincides with the first time the gradient norms exceeded 100. The validation accuracy dropped from 73% to 41% at that point and never fully recovered. This pattern is consistent with the attention mask bug combined with post-norm instability. I also notice that the learning rate was not being warmed up, which likely exacerbated the issue.\n"
        "\n"
        "Recommended changes:\n"
        "1. Expand attention mask from (batch, seq) to (batch, 1, 1, seq)\n"
        "2. Switch to pre-norm LayerNorm\n"
        "3. Add 4000-step linear warmup to the learning rate schedule\n"
        "4. Replace sinusoidal PE with RoPE for length extrapolation\n"
        "\n"
        "These changes should bring the validation accuracy back above 70% within 20 epochs of retraining."
    )
})

dataset.append({
    "id": 15,
    "name": "edge_empty_missing_values",
    "text": (
        "Configuration table:\n"
        "| Parameter | Value | Notes |\n"
        "|-----------|-------|-------|\n"
        "| lr        | 0.001 |       |\n"
        "| epochs    |       | default |\n"
        "| batch     | 32    |       |\n"
        "\n"
        "Summary statistics:\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "This table has only a header row."
    )
})

dataset.append({
    "id": 16,
    "name": "edge_multiple_blank_lines",
    "text": (
        "First section complete.\n"
        "\n"
        "\n"
        "\n"
        "Second section begins here.\n"
        "\n"
        "\n"
        "Third section."
    )
})

dataset.append({
    "id": 17,
    "name": "edge_very_long_paragraph",
    "text": (
        "This is a single very long paragraph that contains multiple sentences but does not end with a colon so it should not produce any child blocks or list attachments. "
        "The quick brown fox jumps over the lazy dog near the bank of the river where the fish swim downstream during the autumn months when the leaves turn golden and crisp. "
        "Scientists have discovered a new species of deep-sea jellyfish that emits bioluminescent pulses in complex patterns suggesting a form of communication previously unknown in cnidarians. "
        "The stock market showed mixed results today with the S&P 500 gaining twelve points while the NASDAQ dropped by twenty-two points amid concerns over interest rate hikes by the Federal Reserve scheduled for next quarter. "
        "Machine learning models continue to improve at a remarkable pace with large language models now capable of generating coherent essays solving mathematical problems and writing computer code across dozens of programming languages. "
        "Ancient ruins discovered in the jungles of Cambodia reveal a previously unknown civilization that may have predated the Angkor Wat temple complex by several centuries according to carbon dating of organic materials found at the excavation site. "
        "The recipe for traditional sourdough bread requires only flour water salt and a live starter culture that must be fed daily for at least seven days before the first bake to develop the characteristic tangy flavor and chewy texture."
    )
})

dataset.append({
    "id": 18,
    "name": "edge_code_blocks_complex",
    "text": (
        "The API returns a JSON response:\n"
        "\n"
        "```json\n"
        "{\n"
        "  \"status\": \"ok\",\n"
        "  \"data\": {\n"
        "    \"users\": [\n"
        "      {\"id\": 1, \"name\": \"Alice\"},\n"
        "      {\"id\": 2, \"name\": \"Bob\"}\n"
        "    ]\n"
        "  }\n"
        "}\n"
        "```\n"
        "\n"
        "The corresponding HTML template:\n"
        "\n"
        "```html\n"
        "<div class=\"container\">\n"
        "  <table id=\"user-table\">\n"
        "    <thead>\n"
        "      <tr><th>ID</th><th>Name</th></tr>\n"
        "    </thead>\n"
        "    <tbody id=\"user-list\">\n"
        "    </tbody>\n"
        "  </table>\n"
        "  <button onclick=\"loadUsers()\">Load</button>\n"
        "</div>\n"
        "```\n"
        "\n"
        "And a quick SQL query to verify:\n"
        "\n"
        "```sql\n"
        "SELECT u.id, u.name, COUNT(o.id) as orders\n"
        "FROM users u\n"
        "LEFT JOIN orders o ON u.id = o.user_id\n"
        "GROUP BY u.id, u.name\n"
        "HAVING COUNT(o.id) > 0;\n"
        "```"
    )
})

with open("/home/taya/Projects/MAG-pal/synthetic_test.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} examples.")
print(f"Saved to /home/taya/Projects/MAG-pal/synthetic_test.json")
