from indexer.parser import parse_file, get_ocr, should_index


def test_parsers_extract_locations(corpus, make_settings):
    root, made = corpus
    s = make_settings([root])
    pdf = parse_file(str(made["quote_pdf"]), s)
    assert [x.start for x in pdf.segments if x.unit == "Page"][-2:] == [1, 2] and "XJ-4471-B" in pdf.segments[-2].text
    docx = parse_file(str(made["shaft_docx"]), s)
    assert any("Stress tensor" in x.label for x in docx.segments)
    assert any("sigma_x" in x.text and "|" in x.text for x in docx.segments)       # table text kept
    pptx = parse_file(str(made["deck"]), s)
    assert [x.start for x in pptx.segments] == [1, 2] and "runway" in pptx.segments[1].text   # notes included
    xl = parse_file(str(made["pump_xlsx"]), s)
    assert any("Efficiency (%): " in x.text for x in xl.segments) and xl.segments[0].label == "Sheet 'Q3 Calc'"
    csvp = parse_file(str(made["contacts_csv"]), s)
    assert "Email: dana.w@corp.com" in csvp.segments[0].text


def test_ocr_on_scanned_pdf_and_image(corpus, make_settings):
    root, made = corpus
    s = make_settings([root])
    if not get_ocr(s).available():
        import pytest; pytest.skip("tesseract not installed")
    sp = parse_file(str(made["scanned_pdf"]), s)
    assert sp.ocr_pages == 1 and "HX-99231" in sp.segments[0].text
    img = parse_file(str(made["receipt_img"]), s)
    assert "GREENLEAF" in img.segments[0].text.upper()


def test_no_ocr_degrades_gracefully(corpus, make_settings):
    root, made = corpus
    s = make_settings([root], ocr_backend="none")
    sp = parse_file(str(made["scanned_pdf"]), s)
    assert sp.segments == [] and any("OCR" in w for w in sp.warnings)


def test_corrupt_file_does_not_raise(tmp_path, make_settings):
    bad = tmp_path / "broken.pdf"; bad.write_bytes(b"%PDF-1.4 garbage")
    r = parse_file(str(bad), make_settings([tmp_path]))
    assert r.segments == [] and r.warnings


def test_should_index_rules(tmp_path, make_settings):
    s = make_settings([tmp_path])
    (tmp_path / "node_modules").mkdir(); (tmp_path / "node_modules" / "a.txt").write_text("x")
    (tmp_path / "ok.txt").write_text("x"); (tmp_path / "a.exe").write_bytes(b"x")
    assert should_index(str(tmp_path / "ok.txt"), s, [str(tmp_path)])
    assert not should_index(str(tmp_path / "node_modules" / "a.txt"), s, [str(tmp_path)])
    assert not should_index(str(tmp_path / "a.exe"), s, [str(tmp_path)])
