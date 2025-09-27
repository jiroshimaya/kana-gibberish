from kanagib.gibberish_generator import BigramGenerator, UnigramGenerator


class TestUnigramGeneratorGenerate:
    def test_正常系(self):
        """
        uv run pytest \
        manual_tests/test_gibbrish_generator.py::TestUnigramGeneratorGenerate:: \
        test_正常系 -v -s
        """
        generator = UnigramGenerator()
        for _ in range(10):
            print(generator.generate(10))


class TestBigramGeneratorGenerate:
    def test_正常系_SEPでストップ(self):
        """
        uv run pytest \
        manual_tests/test_gibbrish_generator.py::TestBigramGeneratorGenerate:: \
        test_正常系_SEPでストップ -v -s
        """
        generator = BigramGenerator()
        for _ in range(10):
            print(generator.generate(10, stop_on_sep=True))

    def test_正常系_SEPでストップしない(self):
        """
        uv run pytest \
        manual_tests/test_gibbrish_generator.py::TestBigramGeneratorGenerate:: \
        test_正常系_SEPでストップしない -v -s
        """
        generator = BigramGenerator()
        for _ in range(10):
            print(generator.generate(10, stop_on_sep=False))

    def test_正常系_SEPを避ける(self):
        """
        uv run pytest \
        manual_tests/test_gibbrish_generator.py::TestBigramGeneratorGenerate:: \
        test_正常系_SEPを避ける -v -s
        """
        generator = BigramGenerator()
        for _ in range(10):
            print(generator.generate(10, avoid_sep=True))
