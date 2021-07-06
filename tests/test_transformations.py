import bootstraping_tools as btt


def test_add_offset():
    augend = 1
    addend = 2
    expected = augend + addend
    obtained = btt.add_offset(augend, addend)
    assert expected == obtained
