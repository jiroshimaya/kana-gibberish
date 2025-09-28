import jamorasep

from kanagib.gibberish_generator import BigramGenerator, UnigramGenerator


class TestUnigramGeneratorGenerate:
    def test_正常系(self):
        generator = UnigramGenerator()
        result = generator.generate(10)
        assert len(jamorasep.parse(result, format="katakana")) == 10

    def test_最初の文字制約_無効な文字で開始しない(self):
        generator = UnigramGenerator()
        invalid_first_chars = {"ン", "ー", "ッ"}
        
        # 複数回テストして統計的に確認
        for _ in range(100):
            result = generator.generate(5)
            first_char = result[0] if result else ""
            assert first_char not in invalid_first_chars, f"生成された文字列 '{result}' が無効な文字 '{first_char}' で開始しました"

    def test_長さ1の場合も制約適用(self):
        generator = UnigramGenerator()
        invalid_first_chars = {"ン", "ー", "ッ"}
        
        # 複数回テストして統計的に確認
        for _ in range(50):
            result = generator.generate(1)
            assert result not in invalid_first_chars, f"長さ1で生成された文字 '{result}' が無効な文字です"


class TestBigramGeneratorGenerate:
    def test_正常系(self):
        generator = BigramGenerator()
        result = generator.generate(10, avoid_sep=True)
        assert len(jamorasep.parse(result, format="katakana")) == 10

    def test_min_lengthパラメータ_正常動作(self):
        generator = BigramGenerator()
        # min_length=5で生成し、避難策として十分な長さを確保
        result = generator.generate(max_length=50, min_length=5, avoid_sep=True)
        mora_count = len(jamorasep.parse(result, format="katakana"))
        assert mora_count >= 5, f"生成されたモーラ数 {mora_count} が最小長 5 未満です"

    def test_min_lengthパラメータ_SEP制約との組み合わせ(self):
        generator = BigramGenerator()
        # min_length=3、stop_on_sep=True（デフォルト）で検証
        # min_length未満では<SEP>が候補から自動的に除外される
        result = generator.generate(max_length=10, min_length=3, stop_on_sep=True)
        mora_count = len(jamorasep.parse(result, format="katakana"))
        assert mora_count >= 3, f"生成されたモーラ数 {mora_count} が最小長 3 未満です"

    def test_min_lengthパラメータ_SEP除外効果(self):
        generator = BigramGenerator()
        # min_length未満では<SEP>が候補から除外されることを統計的にテスト
        results = []
        for _ in range(20):  # 複数回実行して統計的に検証
            result = generator.generate(max_length=20, min_length=5, stop_on_sep=True)
            mora_count = len(jamorasep.parse(result, format="katakana"))
            results.append(mora_count)
        
        # 全ての結果がmin_lengthを満たすことを確認
        for count in results:
            assert count >= 5, f"生成されたモーラ数 {count} が最小長 5 未満です"

    def test_min_lengthパラメータ_異常系_負の値(self):
        generator = BigramGenerator()
        try:
            generator.generate(max_length=10, min_length=-1)
            assert False, "負のmin_lengthで例外が発生しませんでした"
        except ValueError as e:
            assert "Expected positive integer for min_length" in str(e)

    def test_min_lengthパラメータ_異常系_ゼロ(self):
        generator = BigramGenerator()
        try:
            generator.generate(max_length=10, min_length=0)
            assert False, "ゼロのmin_lengthで例外が発生しませんでした"
        except ValueError as e:
            assert "Expected positive integer for min_length" in str(e)

    def test_min_lengthパラメータ_異常系_max_lengthより大きい(self):
        generator = BigramGenerator()
        try:
            generator.generate(max_length=5, min_length=10)
            assert False, "min_length > max_lengthで例外が発生しませんでした"
        except ValueError as e:
            assert "min_length (10) cannot be greater than max_length (5)" in str(e)
