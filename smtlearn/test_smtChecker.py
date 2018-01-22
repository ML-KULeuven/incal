from unittest import TestCase

import pysmt.shortcuts as smt

from smt_check import SmtChecker


class TestSmtChecker(TestCase):
    def test_constant_true(self):
        checker = SmtChecker({})
        self.assertTrue(checker.walk_smt(smt.Bool(True)))

    def test_constant_false(self):
        checker = SmtChecker({})
        self.assertFalse(checker.walk_smt(smt.Bool(False)))

    def test_simpleSymbol_true(self):
        symbol_name = "b"

        checker = SmtChecker({symbol_name: True})
        self.assertTrue(checker.walk_smt(smt.Symbol(symbol_name)))

        checker = SmtChecker({symbol_name: smt.Bool(True)})
        self.assertTrue(checker.walk_smt(smt.Symbol(symbol_name)))

    def test_simpleSymbol_false(self):
        symbol_name = "b"

        checker = SmtChecker({symbol_name: False})
        self.assertFalse(checker.walk_smt(smt.Symbol(symbol_name)))

        checker = SmtChecker({symbol_name: smt.Bool(False)})
        self.assertFalse(checker.walk_smt(smt.Symbol(symbol_name)))
