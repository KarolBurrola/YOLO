"""
Test suite for license plate formatting and validation in util.py.
Validates Sonora (Mexico) formats, ambiguous character corrections, blacklist filtering,
and backward-compatibility with UK format.
"""
from util import format_license, license_complies_format

def test_formats():
    test_cases = [
        # (Input text, Expected output, Description)
        # 1. Sonora standard format: 3 letters + 4 digits
        ("WYZ-12-34", "WYZ1234", "Sonora particular estandar con guiones"),
        ("wyz 1234", "WYZ1234", "Sonora particular minusculas con espacio"),
        ("WYZI234", "WYZ1234", "Sonora particular con I corregida a 1"),
        ("WYZ12O4", "WYZ1204", "Sonora particular con O corregida a 0"),
        ("0YZ1234", "OYZ1234", "Sonora particular con 0 inicial corregido a O"),
        
        # 2. Sonora SCT nuevo formato: 3 letras + 3 digitos + 1 letra
        ("WAB-123-A", "WAB123A", "Sonora SCT nuevo con guiones"),
        ("WAB1234", "WAB1234", "Sonora estandar 7 caracteres"),
        ("WABI23A", "WAB123A", "Sonora SCT con I corregida a 1"),

        # 3. Sonora clasico 6 caracteres: 3 letras + 3 digitos
        ("WAA-123", "WAA123", "Sonora clasico 6 caracteres"),
        ("WAAI23", "WAA123", "Sonora clasico con I corregida a 1"),

        # 4. UK / Demo original (backward compatibility)
        ("CE61 VYL", "CE61VYL", "UK formato original con espacio"),
        ("CE6IVYL", "CE61VYL", "UK formato original con I corregida a 1"),
        ("CE61VYL", "CE61VYL", "UK formato original exacto"),

        # 5. Invalid / Blacklisted
        ("SONORA", None, "Palabra reservada del estado"),
        ("MEXICO", None, "Palabra reservada nacional"),
        ("1234567", None, "Solo digitos (invalido)"),
        ("ABCDEFG", None, "Solo letras (invalido)"),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("EJECUTANDO BANCO DE PRUEBAS PARA PLACAS DE SONORA Y MEXICO")
    print("=" * 60)

    for raw, expected, desc in test_cases:
        actual = format_license(raw)
        valid = license_complies_format(raw)

        expected_valid = expected is not None
        if actual == expected and valid == expected_valid:
            print(f"[OK] {desc}: '{raw}' -> '{actual}'")
            passed += 1
        else:
            print(f"[FALLO] {desc}: '{raw}'")
            print(f"        Esperado: {expected} (valido={expected_valid})")
            print(f"        Obtenido: {actual} (valido={valid})")
            failed += 1

    print("=" * 60)
    print(f"RESULTADOS: {passed} exitosas, {failed} fallidas.")
    print("=" * 60)
    
    return failed == 0

if __name__ == '__main__':
    success = test_formats()
    exit(0 if success else 1)
