"""Deterministic word-internal ARPAbet syllabification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from syllable_recognition.contracts.guided import CONSONANTS, VOWELS


class SyllabificationError(ValueError):
    """Raised when a phone sequence violates the configured syllable rules."""


@dataclass(frozen=True)
class Syllable:
    """一个音节的组合式标签：起始辅音、元音核、结尾辅音和重音。"""

    onset: tuple[str, ...]
    nucleus: str
    coda: tuple[str, ...]
    stress: int

    @property
    def phones(self) -> tuple[str, ...]:
        # 结构中 nucleus 与 stress 分开保存，展开时再拼回 AH0、AE1 等标签。
        return self.onset + (f"{self.nucleus}{self.stress}",) + self.coda

    @property
    def label(self) -> str:
        return "-".join(self.phones)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phones": list(self.phones),
            "label": self.label,
            "onset": list(self.onset),
            "nucleus": self.nucleus,
            "coda": list(self.coda),
            "stress": self.stress,
        }


class Syllabifier:
    """按版本化合法 onset 表执行确定性的单词内部音节化。"""

    def __init__(self, rule_version: str, legal_onsets: set[tuple[str, ...]]) -> None:
        self.rule_version = rule_version
        self.legal_onsets = legal_onsets

    @classmethod
    def from_config(cls, path: Path) -> "Syllabifier":
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if config.get("scope") != "word_internal" or config.get("cross_word_resyllabification"):
            raise SyllabificationError("v1 requires word-internal syllabification")
        if set(config["vowels"]) != VOWELS or set(config["consonants"]) != CONSONANTS:
            raise SyllabificationError("syllabifier phone inventory differs from the public contract")
        legal_onsets = {tuple(cluster) for cluster in config["legal_intervocalic_onsets"]}
        return cls(config["rule_version"], legal_onsets)

    @staticmethod
    def _parse_phone(phone: str) -> tuple[str, int | None]:
        if phone[-1:] in {"0", "1", "2"}:
            base, stress = phone[:-1], int(phone[-1])
            if base not in VOWELS:
                raise SyllabificationError(f"invalid stressed phone {phone!r}")
            return base, stress
        if phone in VOWELS:
            raise SyllabificationError(f"vowel {phone!r} is missing stress")
        if phone not in CONSONANTS:
            raise SyllabificationError(f"unknown ARPAbet phone {phone!r}")
        return phone, None

    def _split_cluster(self, cluster: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        # 最大 onset 原则：优先把最长的合法辅音簇分给后一个音节。
        for onset_length in range(len(cluster), 0, -1):
            onset = cluster[-onset_length:]
            if onset in self.legal_onsets:
                return cluster[:-onset_length], onset
        return cluster, ()

    def syllabify(self, phones: tuple[str, ...] | list[str]) -> tuple[Syllable, ...]:
        """把一个单词的 ARPAbet 音素无损拆成一个或多个音节。"""
        phone_tuple = tuple(phone.upper() for phone in phones)
        parsed = [self._parse_phone(phone) for phone in phone_tuple]
        # 每个带重音元音定义一个音节核，也就确定了音节数量。
        nuclei = [index for index, (base, _) in enumerate(parsed) if base in VOWELS]
        if not nuclei:
            raise SyllabificationError("word pronunciation has no vowel nucleus")

        onsets: list[tuple[str, ...]] = [phone_tuple[: nuclei[0]]]
        codas: list[tuple[str, ...]] = []
        for left, right in zip(nuclei, nuclei[1:]):
            # 两个元音核之间的辅音簇需分给前一音节 coda 和后一音节 onset。
            cluster = phone_tuple[left + 1 : right]
            coda, onset = self._split_cluster(cluster)
            codas.append(coda)
            onsets.append(onset)
        codas.append(phone_tuple[nuclei[-1] + 1 :])

        syllables = []
        for index, nucleus_index in enumerate(nuclei):
            nucleus, stress = parsed[nucleus_index]
            if stress is None:
                raise AssertionError("nucleus stress validation failed")
            syllables.append(Syllable(onsets[index], nucleus, codas[index], stress))

        reconstructed = tuple(phone for syllable in syllables for phone in syllable.phones)
        # 关键门禁：音节化只能重新分组，不能新增、删除或改写任何音素。
        if reconstructed != phone_tuple:
            raise SyllabificationError(
                f"syllabification is not lossless: input={phone_tuple!r}, output={reconstructed!r}"
            )
        return tuple(syllables)

