import jamorasep

from kanagib.gibberish_generator import BigramGenerator, UnigramGenerator


class TestUnigramGeneratorGenerate:
    def test_正常系(self):
        generator = UnigramGenerator()
        result = generator.generate(10)
        assert len(jamorasep.parse(result, format="katakana")) == 10


class TestBigramGeneratorGenerate:
    def test_正常系(self):
        generator = BigramGenerator()
        result = generator.generate(10, avoid_sep=True)
        assert len(jamorasep.parse(result, format="katakana")) == 10
