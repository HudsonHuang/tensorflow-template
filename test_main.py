# -*- coding: utf-8 -*-
import pytest
import main

def test_main():
    assert main.FLAGS==None
    assert main.FLAGS.model==None

if __name__ == '__main__':
    pytest.main()
    