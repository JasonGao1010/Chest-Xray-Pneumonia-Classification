from xray_pneumonia.protocol import Identity, artifact_name, legacy_family, load_protocol


def test_final_protocol_has_stable_unversioned_names():
    protocol = load_protocol()
    assert protocol["protocol_id"] == "CXRShift"
    assert set(protocol["recipes"]) == {"ERM", "ERM-Reg", "JT", "JT-DBS"}
    assert set(protocol["datasets"]) == {"Kermany-FG", "RSNA-1707"}


def test_run_and_artifact_names_are_complete_and_unambiguous():
    identity = Identity("DenseNet121", "JT-DBS", 42)
    assert identity.run_id == "CXRShift__DenseNet121__JT-DBS__s42"
    assert artifact_name(identity, "RSNA-1707", "test", "predictions", "csv") == (
        "CXRShift__DenseNet121__JT-DBS__s42__RSNA-1707__test__predictions.csv"
    )


def test_legacy_result_families_remain_readable():
    assert legacy_family("ERM") == "strict"
    assert legacy_family("ERM-Reg") == "robust"
    assert legacy_family("JT") == "mixed_simple"
    assert legacy_family("JT-DBS") == "mixed_domain_balanced"
