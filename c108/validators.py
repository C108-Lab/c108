"""
C108 Data Validators

This module contains validation functions for data formats and values
including emails, IP addresses, language codes, and URLs.

These validators check the **content/format** of values, not their types.
For runtime type validation, see the abc module.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import ipaddress
import re

from collections.abc import Iterable
from typing import Literal, TypeVar
from urllib.parse import urlparse

# Local ----------------------------------------------------------------------------------------------------------------
from .abc import classgetter
from .tools import fmt_type, fmt_value


# Constants ------------------------------------------------------------------------------------------------------------

T = TypeVar("T")


class LanguageCodes:
    """
    Language and script code constants for validation.

    Provides official code sets from ISO standards for validating language codes
    and writing system (script) identifiers. Used by language validation functions
    to verify codes against international standards.

    Intended for lookup/validation only.

    Attributes:
        ISO_639_1_CODES: Set of valid ISO 639-1 two-letter language codes.
            Covers 184 major world languages with standardized identifiers.
            Standard: ISO 639-1:2002 (https://datahub.io/core/language-codes)
            Examples: {'en', 'fr', 'de', 'ja', 'zh', 'ar', ...}

        ISO_15924_CODES: Set of valid ISO 15924 four-letter script codes.
            Identifies writing systems/scripts (210+ unique scripts).
            Standard: ISO 15924:2022 (https://localizely.com/iso-15924-list/)
            Examples: {'Latn', 'Cyrl', 'Arab', 'Hans', 'Hant', 'Deva', ...}
    """

    ISO_639_1_CODES = {
        "aa",  # Afar
        "ab",  # Abkhazian
        "ae",  # Avestan
        "af",  # Afrikaans
        "ak",  # Akan
        "am",  # Amharic
        "an",  # Aragonese
        "ar",  # Arabic
        "as",  # Assamese
        "av",  # Avaric
        "ay",  # Aymara
        "az",  # Azerbaijani
        "ba",  # Bashkir
        "be",  # Belarusian
        "bg",  # Bulgarian
        "bh",  # Bihari languages
        "bi",  # Bislama
        "bm",  # Bambara
        "bn",  # Bengali
        "bo",  # Tibetan
        "br",  # Breton
        "bs",  # Bosnian
        "ca",  # Catalan
        "ce",  # Chechen
        "ch",  # Chamorro
        "co",  # Corsican
        "cr",  # Cree
        "cs",  # Czech
        "cu",  # Church Slavic
        "cv",  # Chuvash
        "cy",  # Welsh
        "da",  # Danish
        "de",  # German
        "dv",  # Divehi
        "dz",  # Dzongkha
        "ee",  # Ewe
        "el",  # Greek
        "en",  # English
        "eo",  # Esperanto
        "es",  # Spanish
        "et",  # Estonian
        "eu",  # Basque
        "fa",  # Persian
        "ff",  # Fulah
        "fi",  # Finnish
        "fj",  # Fijian
        "fo",  # Faroese
        "fr",  # French
        "fy",  # Western Frisian
        "ga",  # Irish
        "gd",  # Gaelic
        "gl",  # Galician
        "gn",  # Guarani
        "gu",  # Gujarati
        "gv",  # Manx
        "ha",  # Hausa
        "he",  # Hebrew
        "hi",  # Hindi
        "ho",  # Hiri Motu
        "hr",  # Croatian
        "ht",  # Haitian
        "hu",  # Hungarian
        "hy",  # Armenian
        "hz",  # Herero
        "ia",  # Interlingua
        "id",  # Indonesian
        "ie",  # Interlingue
        "ig",  # Igbo
        "ii",  # Sichuan Yi
        "ik",  # Inupiaq
        "io",  # Ido
        "is",  # Icelandic
        "it",  # Italian
        "iu",  # Inuktitut
        "ja",  # Japanese
        "jv",  # Javanese
        "ka",  # Georgian
        "kg",  # Kongo
        "ki",  # Kikuyu
        "kj",  # Kuanyama
        "kk",  # Kazakh
        "kl",  # Kalaallisut
        "km",  # Central Khmer
        "kn",  # Kannada
        "ko",  # Korean
        "kr",  # Kanuri
        "ks",  # Kashmiri
        "ku",  # Kurdish
        "kv",  # Komi
        "kw",  # Cornish
        "ky",  # Kirghiz
        "la",  # Latin
        "lb",  # Luxembourgish
        "lg",  # Ganda
        "li",  # Limburgan
        "ln",  # Lingala
        "lo",  # Lao
        "lt",  # Lithuanian
        "lu",  # Luba-Katanga
        "lv",  # Latvian
        "mg",  # Malagasy
        "mh",  # Marshallese
        "mi",  # Maori
        "mk",  # Macedonian
        "ml",  # Malayalam
        "mn",  # Mongolian
        "mr",  # Marathi
        "ms",  # Malay
        "mt",  # Maltese
        "my",  # Burmese
        "na",  # Nauru
        "nb",  # Norwegian Bokmål
        "nd",  # North Ndebele
        "ne",  # Nepali
        "ng",  # Ndonga
        "nl",  # Dutch
        "nn",  # Norwegian Nynorsk
        "no",  # Norwegian
        "nr",  # South Ndebele
        "nv",  # Navajo
        "ny",  # Chichewa
        "oc",  # Occitan
        "oj",  # Ojibwa
        "om",  # Oromo
        "or",  # Oriya
        "os",  # Ossetian
        "pa",  # Panjabi
        "pi",  # Pali
        "pl",  # Polish
        "ps",  # Pushto
        "pt",  # Portuguese
        "qu",  # Quechua
        "rm",  # Romansh
        "rn",  # Rundi
        "ro",  # Romanian
        "ru",  # Russian
        "rw",  # Kinyarwanda
        "sa",  # Sanskrit
        "sc",  # Sardinian
        "sd",  # Sindhi
        "se",  # Northern Sami
        "sg",  # Sango
        "si",  # Sinhala
        "sk",  # Slovak
        "sl",  # Slovenian
        "sm",  # Samoan
        "sn",  # Shona
        "so",  # Somali
        "sq",  # Albanian
        "sr",  # Serbian
        "ss",  # Swati
        "st",  # Southern Sotho
        "su",  # Sundanese
        "sv",  # Swedish
        "sw",  # Swahili
        "ta",  # Tamil
        "te",  # Telugu
        "tg",  # Tajik
        "th",  # Thai
        "ti",  # Tigrinya
        "tk",  # Turkmen
        "tl",  # Tagalog
        "tn",  # Tswana
        "to",  # Tonga
        "tr",  # Turkish
        "ts",  # Tsonga
        "tt",  # Tatar
        "tw",  # Twi
        "ty",  # Tahitian
        "ug",  # Uighur
        "uk",  # Ukrainian
        "ur",  # Urdu
        "uz",  # Uzbek
        "ve",  # Venda
        "vi",  # Vietnamese
        "vo",  # Volapük
        "wa",  # Walloon
        "wo",  # Wolof
        "xh",  # Xhosa
        "yi",  # Yiddish
        "yo",  # Yoruba
        "za",  # Zhuang
        "zh",  # Chinese
        "zu",  # Zulu
    }
    ISO_15924_CODES = {
        "adlm",  # Adlam
        "afak",  # Afaka
        "aghb",  # Caucasian Albanian
        "ahom",  # Ahom, Tai Ahom
        "arab",  # Arabic
        "aran",  # Arabic (Nastaliq variant)
        "armi",  # Imperial Aramaic
        "armn",  # Armenian
        "avst",  # Avestan
        "bali",  # Balinese
        "bamu",  # Bamum
        "bass",  # Bassa Vah
        "batk",  # Batak
        "beng",  # Bengali (Bangla)
        "bhks",  # Bhaiksuki
        "blis",  # Blissymbols
        "bopo",  # Bopomofo
        "brah",  # Brahmi
        "brai",  # Braille
        "bugi",  # Buginese
        "buhd",  # Buhid
        "cakm",  # Chakma
        "cans",  # Unified Canadian Aboriginal Syllabics
        "cari",  # Carian
        "cham",  # Cham
        "cher",  # Cherokee
        "chrs",  # Chorasmian
        "cirt",  # Cirth
        "copt",  # Coptic
        "cpmn",  # Cypro-Minoan
        "cprt",  # Cypriot syllabary
        "cyrl",  # Cyrillic
        "cyrs",  # Cyrillic (Old Church Slavonic variant)
        "deva",  # Devanagari (Nagari)
        "diak",  # Dives Akuru
        "dogr",  # Dogra
        "dsrt",  # Deseret (Mormon)
        "dupl",  # Duployan shorthand, Duployan stenography
        "egyd",  # Egyptian demotic
        "egyh",  # Egyptian hieratic
        "egyp",  # Egyptian hieroglyphs
        "elba",  # Elbasan
        "elym",  # Elymaic
        "ethi",  # Ethiopic (Geʻez)
        "geok",  # Khutsuri (Asomtavruli and Nuskhuri)
        "geor",  # Georgian (Mkhedruli and Mtavruli)
        "glag",  # Glagolitic
        "gong",  # Gunjala Gondi
        "gonm",  # Masaram Gondi
        "goth",  # Gothic
        "gran",  # Grantha
        "grek",  # Greek
        "gujr",  # Gujarati
        "guru",  # Gurmukhi
        "hanb",  # Han with Bopomofo (alias for Han + Bopomofo)
        "hang",  # Hangul (Hangŭl, Hangeul)
        "hani",  # Han (Hanzi, Kanji, Hanja)
        "hano",  # Hanunoo (Hanunóo)
        "hans",  # Han (Simplified variant)
        "hant",  # Han (Traditional variant)
        "hatr",  # Hatran
        "hebr",  # Hebrew
        "hira",  # Hiragana
        "hluw",  # Anatolian Hieroglyphs (Luwian Hieroglyphs, Hittite Hieroglyphs)
        "hmng",  # Pahawh Hmong
        "hmnp",  # Nyiakeng Puachue Hmong
        "hrkt",  # Japanese syllabaries (alias for Hiragana + Katakana)
        "hung",  # Old Hungarian (Hungarian Runic)
        "inds",  # Indus (Harappan)
        "ital",  # Old Italic (Etruscan, Oscan, etc.)
        "jamo",  # Jamo (alias for Jamo subset of Hangul)
        "java",  # Javanese
        "jpan",  # Japanese (alias for Han + Hiragana + Katakana)
        "jurc",  # Jurchen
        "kali",  # Kayah Li
        "kana",  # Katakana
        "khar",  # Kharoshthi
        "khmr",  # Khmer
        "khoj",  # Khojki
        "kitl",  # Khitan large script
        "kits",  # Khitan small script
        "knda",  # Kannada
        "kore",  # Korean (alias for Hangul + Han)
        "kpel",  # Kpelle
        "kthi",  # Kaithi
        "lana",  # Tai Tham (Lanna)
        "laoo",  # Lao
        "latf",  # Latin (Fraktur variant)
        "latg",  # Latin (Gaelic variant)
        "latn",  # Latin
        "leke",  # Leke
        "lepc",  # Lepcha (Róng)
        "limb",  # Limbu
        "lina",  # Linear A
        "linb",  # Linear B
        "lisu",  # Lisu (Fraser)
        "loma",  # Loma
        "lyci",  # Lycian
        "lydi",  # Lydian
        "mahj",  # Mahajani
        "maka",  # Makasar
        "mand",  # Mandaic, Mandaean
        "mani",  # Manichaean
        "marc",  # Marchen
        "maya",  # Mayan hieroglyphs
        "medf",  # Medefaidrin (Oberi Okaime, Oberi Ɔkaimɛ)
        "mend",  # Mende Kikakui
        "merc",  # Meroitic Cursive
        "mero",  # Meroitic Hieroglyphs
        "mlym",  # Malayalam
        "modi",  # Modi, Moḍī
        "mong",  # Mongolian
        "moon",  # Moon (Moon code, Moon script, Moon type)
        "mroo",  # Mro, Mru
        "mtei",  # Meitei Mayek (Meithei, Meetei)
        "mult",  # Multani
        "mymr",  # Myanmar (Burmese)
        "nand",  # Nandinagari
        "narb",  # Old North Arabian (Ancient North Arabian)
        "nbat",  # Nabataean
        "newa",  # Newa, Newar, Newari, Nepāla lipi
        "nkdb",  # Naxi Dongba (na²¹ɕi³³ to³³ba²¹, Nakhi Tomba)
        "nkgb",  # Naxi Geba (na²¹ɕi³³ gʌ²¹ba²¹, 'Na-'Khi ²Ggŏ-¹baw, Nakhi Geba)
        "nkoo",  # N'Ko
        "nshu",  # Nüshu
        "ogam",  # Ogham
        "olck",  # Ol Chiki (Ol Cemet', Ol, Santali)
        "orkh",  # Old Turkic, Orkhon Runic
        "orya",  # Oriya (Odia)
        "osge",  # Osage
        "osma",  # Osmanya
        "palm",  # Palmyrene
        "pauc",  # Pau Cin Hau
        "perm",  # Old Permic
        "phag",  # Phags-pa
        "phli",  # Inscriptional Pahlavi
        "phlp",  # Psalter Pahlavi
        "phlv",  # Book Pahlavi
        "phnx",  # Phoenician
        "piqd",  # Klingon (KLI pIqaD)
        "plrd",  # Miao (Pollard)
        "prti",  # Inscriptional Parthian
        "qaaa",  # Reserved for private use (start)
        "qabx",  # Reserved for private use (end)
        "rjng",  # Rejang (Redjang, Kaganga)
        "rohg",  # Hanifi Rohingya
        "roro",  # Rongorongo
        "runr",  # Runic
        "samr",  # Samaritan
        "sara",  # Sarati
        "sarb",  # Old South Arabian
        "saur",  # Saurashtra
        "sgnw",  # SignWriting
        "shaw",  # Shavian (Shaw)
        "shrd",  # Sharada, Śāradā
        "shui",  # Shuishu
        "sidd",  # Siddham, Siddhaṃ, Siddhamātṛkā
        "sind",  # Khudawadi, Sindhi
        "sinh",  # Sinhala
        "sogd",  # Sogdian
        "sogo",  # Old Sogdian
        "sora",  # Sora Sompeng
        "soyo",  # Soyombo
        "sund",  # Sundanese
        "sylo",  # Syloti Nagri
        "syrc",  # Syriac
        "syre",  # Syriac (Estrangelo variant)
        "syrj",  # Syriac (Western variant)
        "syrn",  # Syriac (Eastern variant)
        "tagb",  # Tagbanwa
        "takr",  # Takri, Ṭākrī, Ṭāṅkrī
        "tale",  # Tai Le
        "talu",  # New Tai Lue
        "taml",  # Tamil
        "tang",  # Tangut
        "tavt",  # Tai Viet
        "telu",  # Telugu
        "teng",  # Tengwar
        "tfng",  # Tifinagh (Berber)
        "tglg",  # Tagalog (Baybayin, Alibata)
        "thaa",  # Thaana
        "thai",  # Thai
        "tibt",  # Tibetan
        "tirh",  # Tirhuta
        "tnsa",  # Tangsa
        "toto",  # Toto
        "ugar",  # Ugaritic
        "vaii",  # Vai
        "visp",  # Visible Speech
        "vith",  # Vithkuqi
        "wara",  # Warang Citi (Varang Kshiti)
        "wcho",  # Wancho
        "wole",  # Woleai
        "xpeo",  # Old Persian
        "xsux",  # Cuneiform, Sumero-Akkadian
        "yezi",  # Yezidi
        "yiii",  # Yi
        "zanb",  # Zanabazar Square (Zanabazarin Dörböljin Useg, Xewtee Dörböljin Bicig, Horizontal Square Script)
        "zinh",  # Code for inherited script
        "zmth",  # Mathematical notation
        "zsye",  # Symbols (Emoji variant)
        "zsym",  # Symbols
        "zxxx",  # Code for unwritten documents
        "zyyy",  # Code for undetermined script
        "zzzz",  # Code for uncoded script
    }


class CountryCodes:
    """
    Country and region code constants for validation.

    Provides official ISO 3166-1 country codes for validating country/region
    identifiers in addresses, locales, phone numbers, and geographic data.
    Used by validators to verify country codes against international standards.

    Intended for lookup/validation only.

    Attributes:
        ISO_3166_1_CODES: Set of valid ISO 3166-1 alpha-2 country codes.
            Two-letter codes for countries, dependencies, and special areas.
            Standard: ISO 3166-1:2020 [[2]](https://en.wikipedia.org/wiki/ISO_3166-1)
            Examples: {'US', 'GB', 'FR', 'DE', 'JP', 'CN', 'CA', ...}
            Includes territories and special regions (e.g., 'AQ' for Antarctica).
    """

    ISO_3166_1_CODES = {
        "ad",  # Andorra
        "ae",  # United Arab Emirates
        "af",  # Afghanistan
        "ag",  # Antigua and Barbuda
        "ai",  # Anguilla
        "al",  # Albania
        "am",  # Armenia
        "ao",  # Angola
        "aq",  # Antarctica
        "ar",  # Argentina
        "as",  # American Samoa
        "at",  # Austria
        "au",  # Australia
        "aw",  # Aruba
        "ax",  # Åland Islands
        "az",  # Azerbaijan
        "ba",  # Bosnia and Herzegovina
        "bb",  # Barbados
        "bd",  # Bangladesh
        "be",  # Belgium
        "bf",  # Burkina Faso
        "bg",  # Bulgaria
        "bh",  # Bahrain
        "bi",  # Burundi
        "bj",  # Benin
        "bl",  # Saint Barthélemy
        "bm",  # Bermuda
        "bn",  # Brunei Darussalam
        "bo",  # Bolivia
        "bq",  # Bonaire, Sint Eustatius and Saba
        "br",  # Brazil
        "bs",  # Bahamas
        "bt",  # Bhutan
        "bv",  # Bouvet Island
        "bw",  # Botswana
        "by",  # Belarus
        "bz",  # Belize
        "ca",  # Canada
        "cc",  # Cocos (Keeling) Islands
        "cd",  # Congo, Democratic Republic of the
        "cf",  # Central African Republic
        "cg",  # Congo
        "ch",  # Switzerland
        "ci",  # Côte d'Ivoire
        "ck",  # Cook Islands
        "cl",  # Chile
        "cm",  # Cameroon
        "cn",  # China
        "co",  # Colombia
        "cr",  # Costa Rica
        "cu",  # Cuba
        "cv",  # Cabo Verde
        "cw",  # Curaçao
        "cx",  # Christmas Island
        "cy",  # Cyprus
        "cz",  # Czechia
        "de",  # Germany
        "dj",  # Djibouti
        "dk",  # Denmark
        "dm",  # Dominica
        "do",  # Dominican Republic
        "dz",  # Algeria
        "ec",  # Ecuador
        "ee",  # Estonia
        "eg",  # Egypt
        "eh",  # Western Sahara
        "er",  # Eritrea
        "es",  # Spain
        "et",  # Ethiopia
        "fi",  # Finland
        "fj",  # Fiji
        "fk",  # Falkland Islands (Malvinas)
        "fm",  # Micronesia, Federated States of
        "fo",  # Faroe Islands
        "fr",  # France
        "ga",  # Gabon
        "gb",  # United Kingdom
        "gd",  # Grenada
        "ge",  # Georgia
        "gf",  # French Guiana
        "gg",  # Guernsey
        "gh",  # Ghana
        "gi",  # Gibraltar
        "gl",  # Greenland
        "gm",  # Gambia
        "gn",  # Guinea
        "gp",  # Guadeloupe
        "gq",  # Equatorial Guinea
        "gr",  # Greece
        "gs",  # South Georgia and the South Sandwich Islands
        "gt",  # Guatemala
        "gu",  # Guam
        "gw",  # Guinea-Bissau
        "gy",  # Guyana
        "hk",  # Hong Kong
        "hm",  # Heard Island and McDonald Islands
        "hn",  # Honduras
        "hr",  # Croatia
        "ht",  # Haiti
        "hu",  # Hungary
        "id",  # Indonesia
        "ie",  # Ireland
        "il",  # Israel
        "im",  # Isle of Man
        "in",  # India
        "io",  # British Indian Ocean Territory
        "iq",  # Iraq
        "ir",  # Iran, Islamic Republic of
        "is",  # Iceland
        "it",  # Italy
        "je",  # Jersey
        "jm",  # Jamaica
        "jo",  # Jordan
        "jp",  # Japan
        "ke",  # Kenya
        "kg",  # Kyrgyzstan
        "kh",  # Cambodia
        "ki",  # Kiribati
        "km",  # Comoros
        "kn",  # Saint Kitts and Nevis
        "kp",  # Korea, Democratic People's Republic of
        "kr",  # Korea, Republic of
        "kw",  # Kuwait
        "ky",  # Cayman Islands
        "kz",  # Kazakhstan
        "la",  # Lao People's Democratic Republic
        "lb",  # Lebanon
        "lc",  # Saint Lucia
        "li",  # Liechtenstein
        "lk",  # Sri Lanka
        "lr",  # Liberia
        "ls",  # Lesotho
        "lt",  # Lithuania
        "lu",  # Luxembourg
        "lv",  # Latvia
        "ly",  # Libya
        "ma",  # Morocco
        "mc",  # Monaco
        "md",  # Moldova, Republic of
        "me",  # Montenegro
        "mf",  # Saint Martin (French part)
        "mg",  # Madagascar
        "mh",  # Marshall Islands
        "mk",  # North Macedonia
        "ml",  # Mali
        "mm",  # Myanmar
        "mn",  # Mongolia
        "mo",  # Macao
        "mp",  # Northern Mariana Islands
        "mq",  # Martinique
        "mr",  # Mauritania
        "ms",  # Montserrat
        "mt",  # Malta
        "mu",  # Mauritius
        "mv",  # Maldives
        "mw",  # Malawi
        "mx",  # Mexico
        "my",  # Malaysia
        "mz",  # Mozambique
        "na",  # Namibia
        "nc",  # New Caledonia
        "ne",  # Niger
        "nf",  # Norfolk Island
        "ng",  # Nigeria
        "ni",  # Nicaragua
        "nl",  # Netherlands
        "no",  # Norway
        "np",  # Nepal
        "nr",  # Nauru
        "nu",  # Niue
        "nz",  # New Zealand
        "om",  # Oman
        "pa",  # Panama
        "pe",  # Peru
        "pf",  # French Polynesia
        "pg",  # Papua New Guinea
        "ph",  # Philippines
        "pk",  # Pakistan
        "pl",  # Poland
        "pm",  # Saint Pierre and Miquelon
        "pn",  # Pitcairn
        "pr",  # Puerto Rico
        "ps",  # Palestine, State of
        "pt",  # Portugal
        "pw",  # Palau
        "py",  # Paraguay
        "qa",  # Qatar
        "re",  # Réunion
        "ro",  # Romania
        "rs",  # Serbia
        "ru",  # Russian Federation
        "rw",  # Rwanda
        "sa",  # Saudi Arabia
        "sb",  # Solomon Islands
        "sc",  # Seychelles
        "sd",  # Sudan
        "se",  # Sweden
        "sg",  # Singapore
        "sh",  # Saint Helena, Ascension and Tristan da Cunha
        "si",  # Slovenia
        "sj",  # Svalbard and Jan Mayen
        "sk",  # Slovakia
        "sl",  # Sierra Leone
        "sm",  # San Marino
        "sn",  # Senegal
        "so",  # Somalia
        "sr",  # Suriname
        "ss",  # South Sudan
        "st",  # Sao Tome and Principe
        "sv",  # El Salvador
        "sx",  # Sint Maarten (Dutch part)
        "sy",  # Syrian Arab Republic
        "sz",  # Eswatini
        "tc",  # Turks and Caicos Islands
        "td",  # Chad
        "tf",  # French Southern Territories
        "tg",  # Togo
        "th",  # Thailand
        "tj",  # Tajikistan
        "tk",  # Tokelau
        "tl",  # Timor-Leste
        "tm",  # Turkmenistan
        "tn",  # Tunisia
        "to",  # Tonga
        "tr",  # Turkey
        "tt",  # Trinidad and Tobago
        "tv",  # Tuvalu
        "tw",  # Taiwan, Province of China
        "tz",  # Tanzania, United Republic of
        "ua",  # Ukraine
        "ug",  # Uganda
        "um",  # United States Minor Outlying Islands
        "us",  # United States of America
        "uy",  # Uruguay
        "uz",  # Uzbekistan
        "va",  # Holy See (Vatican City State)
        "vc",  # Saint Vincent and the Grenadines
        "ve",  # Venezuela, Bolivarian Republic of
        "vg",  # Virgin Islands, British
        "vi",  # Virgin Islands, U.S.
        "vn",  # Viet Nam
        "vu",  # Vanuatu
        "wf",  # Wallis and Futuna
        "ws",  # Samoa
        "ye",  # Yemen
        "yt",  # Mayotte
        "za",  # South Africa
        "zm",  # Zambia
        "zw",  # Zimbabwe
    }


# Methods --------------------------------------------------------------------------------------------------------------


def validate_categorical(
    value: str,
    categories: set[str] | list[str] | tuple[str, ...],
    case: bool = True,
    strip: bool = True,
) -> str:
    """
    Validate that a value belongs to a set of allowed categorical values.

    Checks that a value belongs to a predefined set of allowed values
    for validating categorical features, enum-like fields, or any
    constrained choice fields in ML pipelines. Catches typos, data corruption,
    or schema violations early before they propagate through the pipeline.

    The function performs membership testing after optionally normalizing the
    value based on case sensitivity and whitespace settings. For performance,
    categories is internally converted to a set for O(1) lookup.

    Args:
        value: The string value to validate. Must not be None.
        categories: Collection of permitted values. Must be non-empty.
            Internally converted to set for efficient lookup.
        case: If True (default), comparison is case-sensitive. If False,
            comparison is case-insensitive and the original value casing
            is preserved in the return.
        strip: If True (default), leading/trailing whitespace is stripped
            from value before validation. The stripped value is returned.

    Returns:
        The validated value. If strip=True, returns the stripped value.
        Otherwise returns the original value unchanged.

    Raises:
        TypeError: If value is not a string, if value is None, or if
            categories contains non-string elements.
        ValueError: If categories is empty, or if value is not in
            categories after normalization.

    Examples:
        Basic validation:
        >>> validate_categorical('red', ['red', 'green', 'blue'])
        'red'

        Case-insensitive validation:
        >>> validate_categorical('RED', ['red', 'green', 'blue'], case=False)
        'RED'

        >>> validate_categorical('yellow', ['red', 'green', 'blue'])
        Traceback (most recent call last):
            ...
        ValueError: Invalid value 'yellow'. Allowed: blue, green, red

        >>> validate_categorical('  red  ', ['red', 'green', 'blue'], strip=False)
        Traceback (most recent call last):
            ...
        ValueError: Invalid value '  red  '. Allowed: blue, green, red
    """
    # Validate input types
    if value is None:
        raise TypeError(f"value must be a string, got {type(None).__name__}")

    if not isinstance(value, str):
        raise TypeError(f"value must be a string, got {type(value).__name__}")

    # Ensure categories is an iterable of strings (but not a string itself)
    if categories is None or isinstance(categories, (bool, int, float)):
        raise TypeError("categories must be iterable")
    if isinstance(categories, str) or not isinstance(categories, Iterable):
        raise TypeError("categories must be iterable")

    categories_list = list(categories)

    if len(categories_list) == 0:
        raise ValueError("categories cannot be empty")

    # Validate all elements are strings
    if not all(isinstance(c, str) for c in categories_list):
        raise TypeError("categories must contain only strings")

    # Convert to set for O(1) lookup
    try:
        categories_set = set(categories)
    except TypeError as e:
        raise TypeError(f"categories must be iterable: {e}") from e

    # Validate all elements are strings
    non_string = next((v for v in categories_set if not isinstance(v, str)), None)
    if non_string is not None:
        raise TypeError(
            f"categories must contain only strings, found {type(non_string).__name__}"
        )

    # Normalize value based on parameters
    validated_value = value.strip() if strip else value

    # Perform validation
    comparison_set = categories_set if case else {v.lower() for v in categories_set}
    comparison_value = validated_value if case else validated_value.lower()

    if comparison_value not in comparison_set:
        categories_str = ", ".join(sorted(categories_set))
        raise ValueError(
            f"Invalid value '{validated_value}'. Allowed: {categories_str}"
        )

    return validated_value


def validate_email(email: str, *, strip: bool = True, lowercase: bool = True) -> str:
    """
    Validate email address format according to simplified RFC 5322 rules.

    This function performs basic email sanity checks for:
        - Non-empty input
        - Valid format (local-part@domain)
        - Length constraints (RFC 5321)
        - Character set compliance

    Note:
        This validation is intentionally simplified for lightweight use cases like
        form prototypes, data preprocessing, unit tests, and educational examples.
        It does NOT cover all RFC 5322 edge cases (quoted strings, comments,
        IP-literals, internationalized domains) or verify deliverability.

        For production systems requiring full RFC compliance, deliverability
        verification, or high-performance validation, use dedicated libraries:
            - `email-validator`: Full RFC compliance with DNS MX lookup support
            - `pydantic[email]`: Rust-backed validation for speed
            - SMTP verification libraries for deliverability checks (slow)

        While RFC 5321 technically allows case-sensitive local parts, virtually
        all modern email providers treat addresses case-insensitively. The default
        behavior normalizes to lowercase per community standards.

    Args:
        email: Email address string to validate.
        strip: If True, strip leading/trailing whitespace before validation.
            Defaults to True.
        lowercase: If True, convert email to lowercase after validation. Defaults
            to True. Recommended for consistency since email addresses are
            case-insensitive in practice.

    Returns:
        The validated email address. Transformations applied in order:
        1. Strip whitespace (if strip=True)
        2. Validate format
        3. Convert to lowercase (if lowercase=True)

    Raises:
        TypeError: If email is not a string.
        ValueError: If email is empty (after stripping if enabled), exceeds length
            limits (254 chars total, 64 for local part), or has invalid format.

    Examples:
        >>> validate_email("User@example.COM")
        'user@example.com'
        >>> validate_email("")
        Traceback (most recent call last):
            ...
        ValueError: email cannot be empty
        >>> validate_email("invalid.email")
        Traceback (most recent call last):
            ...
        ValueError: invalid email format: missing '@' symbol: 'invalid.email'
        >>> validate_email("a" * 65 + "@example.com")
        Traceback (most recent call last):
            ...
        ValueError: email local part exceeds maximum length of 64 characters (got 65)
    """

    # Type check
    if not isinstance(email, str):
        raise TypeError(f"Email must be a string, got {fmt_type(email)}")

    # Handle whitespace
    if strip:
        email = email.strip()
    elif email != email.strip():
        raise ValueError(
            f"invalid email format: contains leading or trailing whitespace: '{email}'"
        )

    # Check for empty
    if not email:
        raise ValueError("email cannot be empty")

    # Check overall length (RFC 5321: 254 chars max)
    if len(email) > 254:
        raise ValueError(
            f"email exceeds maximum length of 254 characters (got {len(email)})"
        )

    # Check for @ symbol
    if "@" not in email:
        raise ValueError(f"invalid email format: missing '@' symbol: '{email}'")

    # Split and validate structure
    parts = email.rsplit("@", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid email format: multiple '@' symbols: '{email}'")

    local_part, domain = parts

    # Validate local part
    if not local_part:
        raise ValueError(f"invalid email format: missing local part: '{email}'")
    if len(local_part) > 64:
        raise ValueError(
            f"email local part exceeds maximum length of 64 characters (got {len(local_part)})"
        )

    # Validate domain
    if not domain:
        raise ValueError(f"invalid email format: missing domain: '{email}'")
    if domain.startswith(".") or domain.endswith("."):
        raise ValueError(
            f"invalid email format: domain cannot start or end with '.': '{email}'"
        )
    if ".." in domain:
        raise ValueError(
            f"invalid email format: domain cannot contain consecutive dots: '{email}'"
        )
    if "." not in domain:
        raise ValueError(
            f"invalid email format: domain must contain at least one dot: '{email}'"
        )

    # Regex pattern for detailed validation (case-insensitive)
    email_pattern = r"^[a-zA-Z0-9][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$"

    # Special case: single character local part
    if len(local_part) == 1:
        email_pattern = r"^[a-zA-Z0-9]@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$"

    if not re.match(email_pattern, email):
        raise ValueError(f"invalid email format: '{email}'")

    # Normalize to lowercase if requested
    if lowercase:
        email = email.lower()

    return email


def validate_ip_address(
    ip: str,
    *,
    strip: bool = True,
    version: Literal[4, 6, "any"] = "any",
    leading_zeros: bool = False,
) -> str:
    """
    Validate IP address format for IPv4 and/or IPv6.

    This function uses Python's `ipaddress` module for standards-compliant
    validation per RFC 791 (IPv4) and RFC 4291 (IPv6).

    Args:
        ip: IP address string to validate.
        strip: If True, strip leading/trailing whitespace before validation.
            Defaults to True.
        version: Required IP version. Use 4 for IPv4 only, 6 for IPv6 only,
            or "any" to accept either. Defaults to "any".
        leading_zeros: If True, allow leading zeros in IPv4 octets
            (e.g., '192.168.001.001'). Defaults to False as leading zeros
            can be ambiguous (octal vs decimal). This parameter only affects
            IPv4; IPv6 leading zeros are always allowed as they are sql
            in hexadecimal notation.

    Returns:
        The validated IP address string. If strip=True, returns the stripped
        version; otherwise returns the original input if valid. The returned
        format is normalized (IPv6 may be compressed per RFC 5952).

    Raises:
        TypeError: If ip is not a string or version is invalid.
        ValueError: If ip is empty (after stripping if enabled), has invalid
            format, or doesn't match the required version.

    Examples:
        IPv4 validation:
        >>> validate_ip_address("192.168.1.1")
        '192.168.1.1'
        >>> validate_ip_address("  10.0.0.1  ")
        '10.0.0.1'
        >>> validate_ip_address("255.255.255.255")
        '255.255.255.255'
        >>> validate_ip_address("0.0.0.0")
        '0.0.0.0'

        IPv6 validation (leading zeros always allowed):
        >>> validate_ip_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        '2001:db8:85a3::8a2e:370:7334'
        >>> validate_ip_address("::1")
        '::1'
        >>> validate_ip_address("fe80::1")
        'fe80::1'
        >>> validate_ip_address("2001:0db8::0001")
        '2001:db8::1'

        Raises errors:
        >>> validate_ip_address("192.168.1")
        Traceback (most recent call last):
            ...
        ValueError: invalid IP address format: '192.168.1'
        >>> validate_ip_address("not.an.ip.address")
        Traceback (most recent call last):
            ...
        ValueError: invalid IP address format: 'not.an.ip.address'
        >>> validate_ip_address("gggg::1")
        Traceback (most recent call last):
            ...
        ValueError: invalid IP address format: 'gggg::1'
    """
    # Type check
    if not isinstance(ip, str):
        raise TypeError(f"IP address must be a string, got {fmt_type(ip)}")

    if version not in (4, 6, "any"):
        raise TypeError(f"version must be 4, 6, or 'any', got {fmt_value(version)}")

    # Handle whitespace
    if strip:
        ip = ip.strip()
    elif ip != ip.strip():
        raise ValueError(
            f"invalid IP address format: contains leading or trailing whitespace: '{ip}'"
        )

    # Check for empty
    if not ip:
        raise ValueError("IP address cannot be empty")

    # Handle leading zeros in IPv4
    # IPv6 leading zeros are standardin hex notation, so no processing needed
    if "." in ip and ":" not in ip:
        # Likely IPv4
        parts = ip.split(".")
        if len(parts) == 4:
            # Check if any part has leading zeros
            has_leading_zeros = any(len(part) > 1 and part[0] == "0" for part in parts)

            if has_leading_zeros:
                if not leading_zeros:
                    # Reject leading zeros
                    raise ValueError(
                        f"invalid IPv4 address: leading zeros not allowed (octal ambiguity): '{ip}'"
                    )
                else:
                    # Strip leading zeros from each octet
                    # Convert each part to int and back to str to remove leading zeros
                    try:
                        normalized_parts = [str(int(part)) for part in parts]
                        ip = ".".join(normalized_parts)
                    except ValueError:
                        # If int() fails, it's not a valid number - let ipaddress handle it
                        pass

    # Validate using ipaddress module
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        raise ValueError(f"invalid IP address format: '{ip}'")

    # Check version requirement
    if version != "any":
        if ip_obj.version != version:
            actual_version = "IPv4" if ip_obj.version == 4 else "IPv6"
            expected_version = "IPv4" if version == 4 else "IPv6"
            raise ValueError(
                f"IP address version mismatch: '{ip}' is {actual_version}, expected {expected_version}"
            )

    # Return normalized format (ipaddress module normalizes both IPv4 and IPv6)
    return str(ip_obj)


def validate_language_code(
    language_code: str,
    allow_iso639_1: bool = True,
    allow_bcp47: bool = True,
    bcp47_parts: Literal[
        "language-region", "language-script", "language-script-region"
    ] = "language-region",
    strict: bool = True,
    case_sensitive: bool = False,
) -> str:
    """
    Validate language code against ISO 639-1 and/or BCP 47 formats.

    Args:
        language_code: The language code to validate. Leading/trailing whitespace is stripped.
        allow_iso639_1: Whether to accept ISO 639-1 two-letter codes (e.g., 'en', 'fr').
        allow_bcp47: Whether to accept BCP 47 language tags (e.g., 'en-US', 'zh-Hans-CN').
        bcp47_parts: Which BCP 47 components to allow:
            - "language-region": language-region only (e.g., 'en-US')
            - "language-script": language-script only (e.g., 'zh-Hans')
            - "language-script-region": full format (e.g., 'zh-Hans-CN')
        strict: If True, validate against official registries. If False, only validate format.
        case_sensitive: Whether validation should be case-sensitive. If False, returns lowercase.

    Returns:
        The validated language code. Normalized to lowercase unless case_sensitive=True.

    Raises:
        TypeError: If language_code is not a string.
        ValueError: If language_code is invalid, empty, or format not allowed.

    Examples:
        >>> validate_language_code('en')
        'en'
        >>> validate_language_code('EN-US')
        'en-us'
        >>> validate_language_code('  zh-Hans-CN  ', bcp47_parts="language-script-region")
        'zh-hans-cn'
        >>> validate_language_code('xx', strict=False)
        'xx'
        >>> validate_language_code('xx-YY', strict=False)
        'xx-yy'
    """
    # Type validation
    if not isinstance(language_code, str):
        raise TypeError(f"language code must be str, got {fmt_type(language_code)}")

    language_code = language_code.strip()

    if not language_code:
        raise ValueError(
            f"language code cannot be empty or whitespace: '{language_code}'"
        )

    # At least one format must be allowed
    if not allow_iso639_1 and not allow_bcp47:
        raise ValueError(
            "at least one format (allow_iso639_1 or allow_bcp47) must be enabled"
        )

    # Normalize case for validation
    lang_lower = language_code.lower()

    # Check ISO 639-1 format
    if re.match(r"^[a-z]{2}$", lang_lower):
        if not allow_iso639_1:
            raise ValueError(
                f"ISO 639-1 format not allowed: '{language_code}'. "
                f"Use BCP 47 format (e.g., 'en-US') instead."
            )

        if strict and lang_lower not in LanguageCodes.ISO_639_1_CODES:
            raise ValueError(f"unknown ISO 639-1 language code: '{language_code}'")

        return lang_lower if not case_sensitive else language_code.strip()

    # Check BCP 47 format
    if "-" in lang_lower:
        if not allow_bcp47:
            raise ValueError(
                f"BCP 47 format not allowed: '{language_code}'. "
                f"Use ISO 639-1 format (e.g., 'en') instead."
            )

        parts = lang_lower.split("-")

        # Validate based on bcp47_parts setting
        if bcp47_parts == "language-region":
            if len(parts) != 2:
                raise ValueError(
                    f"invalid BCP 47 format: '{language_code}'. Expected language-region format (e.g., 'en-US')"
                )

            lang_part, region_part = parts

            # Validate format
            if not re.match(r"^[a-z]{2,3}$", lang_part):
                raise ValueError(f"invalid language part in BCP 47 code: '{lang_part}'")
            if not re.match(r"^[a-z]{2}$", region_part):
                raise ValueError(f"invalid region part in BCP 47 code: '{region_part}'")

            # Strict validation
            if strict:
                if lang_part not in LanguageCodes.ISO_639_1_CODES:
                    raise ValueError(f"unknown language code in BCP 47: '{lang_part}'")
                if region_part not in CountryCodes.ISO_3166_1_CODES:
                    raise ValueError(f"unknown region code in BCP 47: '{region_part}'")

        elif bcp47_parts == "language-script":
            if len(parts) != 2:
                raise ValueError(
                    f"invalid BCP 47 format: '{language_code}'. Expected language-script format (e.g., 'zh-Hans')"
                )

            lang_part, script_part = parts

            # Validate format
            if not re.match(r"^[a-z]{2,3}$", lang_part):
                raise ValueError(f"invalid language part in BCP 47 code: '{lang_part}'")
            if not re.match(r"^[a-z]{4}$", script_part):
                raise ValueError(f"invalid script part in BCP 47 code: '{script_part}'")

            # Strict validation
            if strict:
                if lang_part not in LanguageCodes.ISO_639_1_CODES:
                    raise ValueError(f"unknown language code in BCP 47: '{lang_part}'")
                if script_part not in LanguageCodes.ISO_15924_CODES:
                    raise ValueError(f"unknown script code in BCP 47: '{script_part}'")

        elif bcp47_parts == "language-script-region":
            if len(parts) != 3:
                raise ValueError(
                    f"invalid BCP 47 format: '{language_code}'. Expected language-script-region format (e.g., 'zh-Hans-CN')"
                )

            lang_part, script_part, region_part = parts

            # Validate format
            if not re.match(r"^[a-z]{2,3}$", lang_part):
                raise ValueError(f"invalid language part in BCP 47 code: '{lang_part}'")
            if not re.match(r"^[a-z]{4}$", script_part):
                raise ValueError(f"invalid script part in BCP 47 code: '{script_part}'")
            if not re.match(r"^[a-z]{2}$", region_part):
                raise ValueError(f"invalid region part in BCP 47 code: '{region_part}'")

            # Strict validation
            if strict:
                if lang_part not in LanguageCodes.ISO_639_1_CODES:
                    raise ValueError(f"unknown language code in BCP 47: '{lang_part}'")
                if script_part not in LanguageCodes.ISO_15924_CODES:
                    raise ValueError(f"unknown script code in BCP 47: '{script_part}'")
                if region_part not in CountryCodes.ISO_3166_1_CODES:
                    raise ValueError(f"unknown region code in BCP 47: '{region_part}'")

        return lang_lower if not case_sensitive else language_code.strip()

    # Invalid format
    allowed_formats = []
    if allow_iso639_1:
        allowed_formats.append("ISO 639-1 format (e.g., 'en', 'fr')")
    if allow_bcp47:
        if bcp47_parts == "language-region":
            allowed_formats.append(
                "BCP 47 language-region format (e.g., 'en-US', 'fr-CA')"
            )
        elif bcp47_parts == "language-script":
            allowed_formats.append(
                "BCP 47 language-script format (e.g., 'zh-Hans', 'sr-Cyrl')"
            )
        elif bcp47_parts == "language-script-region":
            allowed_formats.append(
                "BCP 47 language-script-region format (e.g., 'zh-Hans-CN', 'sr-Cyrl-RS')"
            )

    formats_str = " or ".join(allowed_formats)
    raise ValueError(
        f"invalid language code: '{language_code}'. Expected {formats_str}"
    )


def validate_not_empty(collection: T, *, name: str = "collection") -> T:
    """
    Validate that a collection contains at least one element.

    Ensures a collection (list, tuple, set, dict, NumPy array, Pandas DataFrame/Series,
    JAX array, TensorFlow tensor, PyTorch tensor, etc.) contains at least one element.
    Prevents silent failures from empty batches, missing data, or incorrectly filtered
    datasets that would cause downstream errors in ML pipelines or data processing.

    This function uses stdlib-only dependencies but intelligently detects and validates
    common ML framework data types through duck-typing and attribute inspection.

    Args:
        collection: Any collection type. Supported types include:
            - Standard collections: list, tuple, set, dict, frozenset
            - NumPy arrays (via shape/size attributes)
            - Pandas DataFrame/Series (via empty attribute)
            - JAX arrays (via shape/size attributes)
            - TensorFlow tensors (via shape attribute)
            - PyTorch tensors (via shape/numel attributes)
            - Any sized object with __len__

            Note: Generators and lazy iterators are NOT supported as they cannot be
            validated without consumption. Materialize them first (e.g., list(gen)).
        name: Optional name for the collection used in error messages.
            Defaults to "collection".

    Returns:
        The original collection unchanged if it contains at least one element.

    Raises:
        TypeError: If collection is None, not a supported collection type
            (e.g., int, float, non-collection objects), a string, or a generator/iterator.
        ValueError: If collection is empty (contains zero elements).

    Examples:
        Standard collections:

        >>> validate_not_empty([1, 2, 3])
        [1, 2, 3]
        >>> validate_not_empty({"key": "value"})
        {'key': 'value'}
        >>> validate_not_empty((1,))
        (1,)
        >>> validate_not_empty({1, 2, 3})
        {1, 2, 3}

        Edge cases with empty collections:

        >>> validate_not_empty([])
        Traceback (most recent call last):
            ...
        ValueError: collection must not be empty

        >>> validate_not_empty({}, name="user_data")
        Traceback (most recent call last):
            ...
        ValueError: user_data must not be empty

        >>> validate_not_empty(set())
        Traceback (most recent call last):
            ...
        ValueError: collection must not be empty

        Type validation:

        >>> validate_not_empty(None)
        Traceback (most recent call last):
            ...
        TypeError: collection cannot be None

        >>> validate_not_empty(42)
        Traceback (most recent call last):
            ...
        TypeError: collection must be a collection type, got int

        >>> validate_not_empty("string")
        Traceback (most recent call last):
            ...
        TypeError: strings are not supported as collections

        Generators are rejected:

        >>> validate_not_empty(x for x in range(5))
        Traceback (most recent call last):
            ...
        TypeError: generators and iterators are not supported, materialize to a collection first

        >>> validate_not_empty(iter([1, 2, 3]))
        Traceback (most recent call last):
            ...
        TypeError: generators and iterators are not supported, materialize to a collection first

        Simulating NumPy array behavior:

        >>> class MockArray:
        ...     def __init__(self, shape):
        ...         self.shape = shape
        ...         self.size = 1
        ...         for dim in shape:
        ...             self.size *= dim
        >>> validate_not_empty(MockArray((3, 3)))
        <...MockArray object at 0x...>
        >>> validate_not_empty(MockArray((0, 3)))
        Traceback (most recent call last):
            ...
        ValueError: collection must not be empty

        Simulating Pandas DataFrame behavior:

        >>> class MockDataFrame:
        ...     def __init__(self, empty):
        ...         self.empty = empty
        >>> validate_not_empty(MockDataFrame(False))
        <...MockDataFrame object at 0x...>
        >>> validate_not_empty(MockDataFrame(True))
        Traceback (most recent call last):
            ...
        ValueError: collection must not be empty

        Simulating PyTorch tensor behavior:

        >>> class MockTensor:
        ...     def __init__(self, shape):
        ...         self.shape = shape
        ...     def numel(self):
        ...         result = 1
        ...         for dim in self.shape:
        ...             result *= dim
        ...         return result
        >>> validate_not_empty(MockTensor((2, 3)))
        <...MockTensor object at 0x...>
        >>> validate_not_empty(MockTensor((0, 3)))
        Traceback (most recent call last):
            ...
        ValueError: collection must not be empty
    """
    # Check for None explicitly
    if collection is None:
        raise TypeError("collection cannot be None")

    # Reject strings (they're iterable but usually not intended as collections)
    if isinstance(collection, (str, bytes)):
        raise TypeError("strings are not supported as collections")

    # Check if it's a supported collection type
    is_collection = False
    is_empty = True

    # Method 1: Check for Pandas DataFrame/Series (has .empty attribute)
    if hasattr(collection, "empty") and isinstance(getattr(collection, "empty"), bool):
        is_collection = True
        is_empty = collection.empty

    # Method 2: Check for PyTorch tensors (has .shape and .numel())
    # IMPORTANT: Check this BEFORE NumPy because PyTorch has both .shape and .size
    elif (
        hasattr(collection, "shape")
        and hasattr(collection, "numel")
        and callable(collection.numel)
    ):
        is_collection = True
        try:
            is_empty = collection.numel() == 0
        except Exception:
            # If numel() fails, fall through to other methods
            is_collection = False

    # Method 3: Check for NumPy/JAX/TensorFlow arrays (has .shape and .size attribute, not method)
    elif (
        hasattr(collection, "shape")
        and hasattr(collection, "size")
        and not callable(getattr(collection, "size"))
    ):
        is_collection = True
        # Check if size is 0 or any dimension in shape is 0
        size = getattr(collection, "size")
        is_empty = size == 0

    # Method 4: Standard collections with __len__
    elif hasattr(collection, "__len__"):
        is_collection = True
        try:
            is_empty = len(collection) == 0
        except TypeError as e:
            # Some objects have __len__ but it raises TypeError
            raise TypeError(
                f"collection must be a collection type, got {type(collection).__name__}"
            ) from e

    # Method 5: Reject generators and iterators
    # Check for __iter__ without __len__ (generators, iterators, etc.)
    elif hasattr(collection, "__iter__"):
        # This is likely a generator or iterator - reject it
        raise TypeError(
            "generators and iterators are not supported, materialize to a collection first"
        )

    if not is_collection:
        raise TypeError(
            f"collection must be a collection type, got {type(collection).__name__}"
        )

    if is_empty:
        raise ValueError(f"{name} must not be empty")

    return collection


def validate_positive(value: float, strict: bool = True) -> float:
    """
    Validate value is positive (optionally allowing zero).

    Checks that a numeric value is greater than zero (or greater than/equal to zero if strict=False).
    Common for counts, distances, rates, durations, and other naturally positive quantities in
    ML pipelines. Use strict=False when zero is a valid value (e.g., counts can be zero).

    Args:
        value: The numeric value to validate
        strict: If True, value must be > 0; if False, value must be >= 0

    Returns:
        float: The original value if valid

    Raises:
        ValueError: If value violates the positivity constraint
    """


# Data Structure Validators


def validate_probability(value: float) -> float:
    """
    Validate value is a valid probability between 0 and 1 (inclusive).

    Ensures a numeric value falls within the [0, 1] interval, which is required for probabilities,
    confidence scores, model outputs from sigmoid/softmax layers, mixing weights, and similar
    ML/statistical values. This is a specialized case of range validation optimized for the
    common probability constraint.

    Args:
        value: The numeric value to validate as a probability

    Returns:
        float: The original value if valid

    Raises:
        ValueError: If value is not in [0, 1]
    """


def validate_range(
    value: float, min_value: float | None = None, max_value: float | None = None
) -> float:
    """
    Validate numeric value falls within specified bounds.

    Checks that a numeric value is within the specified minimum and maximum bounds (inclusive).
    Useful for validating ML features, hyperparameters, or any numeric data with known constraints.
    At least one bound (min or max) should be specified.

    Args:
        value: The numeric value to validate
        min_value: Minimum allowed value (inclusive), or None for no lower bound
        max_value: Maximum allowed value (inclusive), or None for no upper bound

    Returns:
        float: The original value if valid

    Raises:
        ValueError: If value is outside the specified range
    """


def validate_shape(array, expected_shape: tuple[int | None, ...]) -> any:
    """
    Validate array has expected shape/dimensions.

    Checks that an array-like object (numpy array, tensor, nested list) matches the expected shape.
    Use None in expected_shape for dimensions that can be any size (e.g., (None, 3) accepts any
    number of rows with 3 columns). Essential for validating inputs/outputs in neural networks
    and matrix operations.

    Args:
        array: Array-like object with a .shape attribute or nested structure
        expected_shape: Tuple of expected dimensions, use None for flexible dimensions

    Returns:
        The original array if shape matches

    Raises:
        ValueError: If shape doesn't match expected_shape
    """


def validate_timestamp(timestamp_str: str, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Validate timestamp string matches expected format and is parseable.

    Checks that a timestamp string conforms to the specified format using strftime/strptime
    conventions and represents a valid datetime. Handles full date+time validation in one step,
    which is the most common temporal data format in ML pipelines (logs, events, time-series).
    Supports ISO8601, custom formats, and can validate Unix timestamps with format="unix".

    Args:
        timestamp_str: The timestamp string to validate
        format: strftime/strptime format string (default: "%Y-%m-%d %H:%M:%S")
                Use "unix" for Unix timestamp (seconds since epoch)

    Returns:
        str: The original timestamp string if valid and parseable

    Raises:
        ValueError: If timestamp doesn't match format or represents invalid datetime
    """


def validate_timestamp_range(
    timestamp_str: str,
    start: str | None = None,
    end: str | None = None,
    format: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Validate timestamp falls within specified datetime range.

    Checks that a timestamp (after parsing) falls between start and end datetimes (inclusive).
    Essential for validating temporal boundaries in time-series data, filtering event logs,
    or ensuring train/test temporal splits. All timestamps must use the same format string.

    Args:
        timestamp_str: The timestamp string to validate
        start: Minimum allowed timestamp (inclusive), or None for no lower bound
        end: Maximum allowed timestamp (inclusive), or None for no upper bound
        format: strftime/strptime format string (default: "%Y-%m-%d %H:%M:%S")

    Returns:
        str: The original timestamp string if within range

    Raises:
        ValueError: If timestamp is outside range or parsing fails
    """


def validate_unique(collection, key=None) -> any:
    """
    Validate all elements in collection are unique.

    Ensures no duplicate values exist in a collection, which is critical for unique identifiers,
    primary keys, index values, or when building lookup structures. For collections of objects,
    use the key parameter to specify which attribute should be unique (similar to sorted/max).

    Args:
        collection: Iterable collection to check for uniqueness
        key: Optional function to extract comparison value from each element

    Returns:
        The original collection if all elements are unique

    Raises:
        ValueError: If duplicate elements are found
    """


# validate_uri() Family of Classes and Methods -------------------------------------------------------------------------


class SchemeGroup:
    """Base class for URI scheme groups."""

    @classgetter(cache=True)
    def all(cls) -> tuple[str, ...]:
        """
        Get all schemes in this group.

        Append all schemes from nested SchemeGroup instances recursively
        """
        schemes = []
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name != "all":
                attr = getattr(cls, attr_name)
                if isinstance(attr, str):
                    schemes.append(attr)
                elif isinstance(attr, type) and issubclass(attr, SchemeGroup):
                    schemes.extend(attr.all)
        return tuple(schemes)


class AnalyticalSchemes(SchemeGroup):
    """Analytical/OLAP database URI schemes."""

    clickhouse = "clickhouse"
    databricks = "databricks"
    druid = "druid"
    impala = "impala"
    presto = "presto"
    snowflake = "snowflake"
    trino = "trino"
    vertica = "vertica"


class AWSDatabaseSchemes(SchemeGroup):
    """AWS managed database URI schemes."""

    athena = "athena"  # Serverless query service
    aurora = "aurora"  # Aurora MySQL/PostgreSQL
    documentdb = "documentdb"  # MongoDB-compatible
    dynamodb = "dynamodb"  # NoSQL key-value
    neptune_db = "neptune-db"  # Graph database
    rds = "rds"  # Relational Database Service
    redshift = "redshift"  # Data warehouse
    timestream = "timestream"  # Time series database


class AWSStorageSchemes(SchemeGroup):
    """AWS S3 URI schemes."""

    s3 = "s3"
    s3a = "s3a"
    s3n = "s3n"


class AzureDatabaseSchemes(SchemeGroup):
    """Azure managed database URI schemes."""

    azuresql = "azuresql"  # Azure SQL Database
    cosmosdb = "cosmosdb"  # Multi-model NoSQL
    sqldw = "sqldw"  # SQL Data Warehouse (legacy name)
    synapse = "synapse"  # Analytics platform (formerly SQL DW)


class AzureStorageSchemes(SchemeGroup):
    """Microsoft Azure storage URI schemes."""

    abfs = "abfs"
    abfss = "abfss"
    adl = "adl"
    az = "az"
    wasb = "wasb"
    wasbs = "wasbs"


class DataVersioningSchemes(SchemeGroup):
    """Data versioning system URI schemes."""

    dvc = "dvc"  # DVC (Data Version Control)
    pachyderm = "pachyderm"  # Pachyderm data pipelines


class DistributedSchemes(SchemeGroup):
    """Distributed file system URI schemes."""

    alluxio = "alluxio"
    ceph = "ceph"
    dbfs = "dbfs"
    minio = "minio"
    rados = "rados"
    swift = "swift"


class GCPDatabaseSchemes(SchemeGroup):
    """GCP managed database URI schemes."""

    bigquery = "bigquery"  # Data warehouse
    bigtable = "bigtable"  # NoSQL wide-column
    datastore = "datastore"  # NoSQL document database (legacy)
    firestore = "firestore"  # NoSQL document database
    spanner = "spanner"  # Distributed SQL database


class GCPStorageSchemes(SchemeGroup):
    """Google Cloud Platform URI schemes."""

    gs = "gs"


class GraphSchemes(SchemeGroup):
    """Graph database URI schemes."""

    arangodb = "arangodb"
    janusgraph = "janusgraph"
    neo4j = "neo4j"
    neo4js = "neo4js"  # Neo4j with encryption
    orientdb = "orientdb"


class HadoopSchemes(SchemeGroup):
    """Hadoop ecosystem URI schemes."""

    hdfs = "hdfs"
    hive = "hive"
    webhdfs = "webhdfs"


class LakehouseSchemes(SchemeGroup):
    """Data lakehouse URI schemes."""

    delta = "delta"
    iceberg = "iceberg"


class LocalSchemes(SchemeGroup):
    """Local and URN schemes."""

    file = "file"
    urn = "urn"


class MLDatasetSchemes(SchemeGroup):
    """ML dataset URI schemes."""

    tfds = "tfds"  # TensorFlow Datasets
    torch = "torch"  # PyTorch datasets


class MLFlowSchemes(SchemeGroup):
    """MLflow-specific URI schemes."""

    models = "models"  # Model Registry: models:/<name>/<version_or_stage>
    runs = "runs"  # Artifact from run: runs:/<run_id>/path


class MLHubSchemes(SchemeGroup):
    """ML model hub URI schemes."""

    hf = "hf"  # Hugging Face Hub
    huggingface = "huggingface"  # Hugging Face Hub (alias)
    onnx = "onnx"  # ONNX Model Zoo
    tfhub = "tfhub"  # TensorFlow Hub
    torchhub = "torchhub"  # PyTorch Hub


class MLTrackingSchemes(SchemeGroup):
    """ML experiment tracking platform URI schemes."""

    aim = "aim"  # Aim
    clearml = "clearml"  # ClearML (formerly Allegro)
    comet = "comet"  # Comet ML
    mlflow = "mlflow"  # MLflow artifacts (generic)
    neptune = "neptune"  # Neptune.ai
    sacred = "sacred"  # Sacred
    tensorboard = "tensorboard"  # TensorBoard logs
    wandb = "wandb"  # Weights & Biases


class NetworkFSSchemes(SchemeGroup):
    """Network file system URI schemes."""

    afp = "afp"
    cifs = "cifs"
    nfs = "nfs"
    smb = "smb"


class NoSQLSchemes(SchemeGroup):
    """NoSQL database URI schemes."""

    cassandra = "cassandra"
    couchbase = "couchbase"
    couchdb = "couchdb"
    cql = "cql"  # Cassandra Query Language
    memcached = "memcached"
    mongo = "mongo"  # Alternative MongoDB scheme
    mongodb = "mongodb"
    redis = "redis"
    rediss = "rediss"  # Redis with SSL/TLS


class SearchSchemes(SchemeGroup):
    """Search and vector database URI schemes."""

    elasticsearch = "elasticsearch"
    es = "es"  # Elasticsearch alias
    meilisearch = "meilisearch"
    opensearch = "opensearch"
    solr = "solr"
    typesense = "typesense"


class SQLSchemes(SchemeGroup):
    """SQL database URI schemes."""

    cockroach = "cockroach"
    cockroachdb = "cockroachdb"
    db2 = "db2"
    mariadb = "mariadb"
    mssql = "mssql"
    mysql = "mysql"
    oracle = "oracle"
    postgres = "postgres"
    postgresql = "postgresql"
    sqlite = "sqlite"
    sqlserver = "sqlserver"
    teradata = "teradata"


class TimeSeriesSchemes(SchemeGroup):
    """Time series database URI schemes."""

    influxdb = "influxdb"
    prometheus = "prometheus"
    timescaledb = "timescaledb"
    victoriametrics = "victoriametrics"


class VectorSchemes(SchemeGroup):
    """Vector database URI schemes (for ML embeddings)."""

    chroma = "chroma"
    chromadb = "chromadb"
    milvus = "milvus"
    pinecone = "pinecone"
    qdrant = "qdrant"
    weaviate = "weaviate"


class WebSchemes(SchemeGroup):
    """Web protocol URI schemes."""

    ftp = "ftp"
    ftps = "ftps"
    http = "http"
    https = "https"


class Scheme:
    """URI scheme definitions organized by category.

    Provides categorized access to all supported URI schemes for cloud storage,
    distributed systems, ML platforms, experiment tracking, and databases.

    Examples:
        >>> # Cloud storage
        >>> Scheme.aws.s3
        's3'

        >>> # ML experiment tracking
        >>> Scheme.ml.tracking.wandb
        'wandb'

        >>> # MLflow-specific
        >>> Scheme.ml.mlflow.runs
        'runs'

        >>> # Model hubs
        >>> Scheme.ml.hub.hf
        'hf'

        >>> # Cloud databases
        >>> Scheme.db.cloud.aws.bigquery
        Traceback (most recent call last):
        ...
        AttributeError: type object 'AWSDatabaseSchemes' has no attribute 'bigquery'

        >>> # Cloud databases (corrected)
        >>> Scheme.db.cloud.gcp.bigquery
        'bigquery'

        >>> # Vector databases
        >>> Scheme.db.vector.pinecone
        'pinecone'

        >>> # Get all database schemes
        >>> schemes = Scheme.db.all
        >>> 'bigquery' in schemes and 'redis' in schemes
        True
    """

    # Cloud providers (storage)
    aws = AWSStorageSchemes
    azure = AzureStorageSchemes
    gcp = GCPStorageSchemes

    # Distributed systems
    distributed = DistributedSchemes
    hadoop = HadoopSchemes
    lakehouse = LakehouseSchemes
    network = NetworkFSSchemes

    # ML/AI platforms (nested for organization)
    class ml:
        """ML/AI platform schemes organized by category."""

        data_versioning = DataVersioningSchemes
        datasets = MLDatasetSchemes
        hub = MLHubSchemes
        mlflow = MLFlowSchemes
        tracking = MLTrackingSchemes

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """Get all ML-related schemes.

            Returns:
                tuple[str, ...]: All ML platform, tracking, hub, and dataset schemes.

            Examples:
                >>> schemes = Scheme.ml.all
                >>> 'wandb' in schemes and 'hf' in schemes and 'runs' in schemes
                True
            """
            return (
                *DataVersioningSchemes.all,
                *MLDatasetSchemes.all,
                *MLFlowSchemes.all,
                *MLHubSchemes.all,
                *MLTrackingSchemes.all,
            )

    # Databases (comprehensive organization)
    class db:
        """Database schemes organized by category."""

        # SQL databases
        sql = SQLSchemes

        # Cloud-managed databases
        class cloud:
            """Cloud-managed database schemes."""

            aws = AWSDatabaseSchemes
            azure = AzureDatabaseSchemes
            gcp = GCPDatabaseSchemes

            @classgetter(cache=True)
            def all(cls) -> tuple[str, ...]:
                """Get all cloud-managed database schemes."""
                return (
                    *AWSDatabaseSchemes.all,
                    *AzureDatabaseSchemes.all,
                    *GCPDatabaseSchemes.all,
                )

        # Database types
        analytical = AnalyticalSchemes
        graph = GraphSchemes
        nosql = NoSQLSchemes
        search = SearchSchemes
        timeseries = TimeSeriesSchemes
        vector = VectorSchemes

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """Get all database schemes.

            Returns:
                tuple[str, ...]: All database schemes including cloud, NoSQL,
                    vector, time series, graph, analytical, and sql databases.

            Examples:
                >>> schemes = Scheme.db.all
                >>> all(s in schemes for s in ['bigquery', 'redis', 'pinecone', 'neo4j'])
                True
            """
            return (
                *AWSDatabaseSchemes.all,
                *AnalyticalSchemes.all,
                *AzureDatabaseSchemes.all,
                *GCPDatabaseSchemes.all,
                *GraphSchemes.all,
                *NoSQLSchemes.all,
                *SQLSchemes.all,
                *SearchSchemes.all,
                *TimeSeriesSchemes.all,
                *VectorSchemes.all,
            )

    # Web and local
    local = LocalSchemes
    web = WebSchemes

    @staticmethod
    def cloud() -> tuple[str, ...]:
        """Get all major cloud provider schemes (AWS, GCP, Azure storage)."""
        return (
            *AWSStorageSchemes.all,
            *AzureStorageSchemes.all,
            *GCPStorageSchemes.all,
        )

    @staticmethod
    def bigdata() -> tuple[str, ...]:
        """Get all big data / distributed system schemes."""
        return (
            *DistributedSchemes.all,
            *HadoopSchemes.all,
            *LakehouseSchemes.all,
        )

    @classgetter(cache=True)
    def all(cls) -> tuple[str, ...]:
        """Get all supported URI schemes."""
        return (
            *AWSDatabaseSchemes.all,
            *AWSStorageSchemes.all,
            *AnalyticalSchemes.all,
            *AzureDatabaseSchemes.all,
            *AzureStorageSchemes.all,
            *DataVersioningSchemes.all,
            *DistributedSchemes.all,
            *GCPDatabaseSchemes.all,
            *GCPStorageSchemes.all,
            *GraphSchemes.all,
            *HadoopSchemes.all,
            *LakehouseSchemes.all,
            *LocalSchemes.all,
            *MLDatasetSchemes.all,
            *MLFlowSchemes.all,
            *MLHubSchemes.all,
            *MLTrackingSchemes.all,
            *NetworkFSSchemes.all,
            *NoSQLSchemes.all,
            *SQLSchemes.all,
            *SearchSchemes.all,
            *TimeSeriesSchemes.all,
            *VectorSchemes.all,
            *WebSchemes.all,
        )


def validate_uri(
    uri: str,
    schemes: list[str] | tuple[str, ...] | None = None,
    allow_query: bool = False,
    allow_relative: bool = False,
    cloud_names: bool = True,
    max_length: int = 8192,
    require_host: bool = True,
) -> str:
    """Validate URI format and structure.

    Validates Uniform Resource Identifiers including URLs, cloud storage URIs,
    distributed file systems, ML platform URIs (MLflow, Weights & Biases,
    Hugging Face), experiment tracking, model registries, and comprehensive
    database schemes (cloud databases, NoSQL, vector DBs, graph DBs, etc.).
    Designed for ML/DS workflows where data and artifacts span multiple storage
    backends and database systems.

    Args:
        uri: The URI string to validate. Leading/trailing whitespace is stripped.
        schemes: Tuple of permitted URI schemes. If None, allows common
            schemes for ML/DS workflows. Use `Scheme` class for organized access:

            **Cloud Storage:**
                - `Scheme.aws.all` - AWS S3 (s3, s3a, s3n)
                - `Scheme.azure.all` - Azure (wasbs, abfs, etc.)
                - `Scheme.gcp.all` - GCP (gs, gcs)
                - `Scheme.cloud()` - All major cloud providers

            **Databases:**
                - `Scheme.db.all` - All database schemes
                - `Scheme.db.cloud.all` - Cloud-managed databases (AWS, GCP, Azure)
                - `Scheme.db.cloud.aws.all` - AWS databases (redshift, dynamodb, athena, etc.)
                - `Scheme.db.cloud.gcp.all` - GCP databases (bigquery, bigtable, spanner, etc.)
                - `Scheme.db.cloud.azure.all` - Azure databases (cosmosdb, synapse, etc.)
                - `Scheme.db.nosql.all` - NoSQL databases (mongodb, redis, cassandra, etc.)
                - `Scheme.db.vector.all` - Vector databases (pinecone, weaviate, qdrant, etc.)
                - `Scheme.db.search.all` - Search databases (elasticsearch, opensearch, etc.)
                - `Scheme.db.timeseries.all` - Time series databases (influxdb, prometheus, etc.)
                - `Scheme.db.graph.all` - Graph databases (neo4j, arangodb, etc.)
                - `Scheme.db.analytical.all` - Analytical databases (clickhouse, snowflake, etc.)
                - `Scheme.db.sql.all` - SQL databases (sqlite, mysql, postgresql)

            **Distributed Systems:**
                - `Scheme.hadoop.all` - Hadoop (hdfs, webhdfs, hive)
                - `Scheme.bigdata()` - All big data systems
                - `Scheme.distributed.all` - Alluxio, Ceph, MinIO, etc.

            **Local:**
                - `Scheme.local.all` - local and URN schemes

            **ML Platforms:**
                - `Scheme.ml.all` - All ML-related schemes
                - `Scheme.ml.mlflow.all` - MLflow (runs, models)
                - `Scheme.ml.tracking.all` - Experiment tracking (wandb, comet, neptune, clearml)
                - `Scheme.ml.model_hub.all` - Model hubs (hf, torchhub, tfhub, onnx)
                - `Scheme.ml.data_versioning.all` - Data versioning (dvc, pachyderm)
                - `Scheme.ml.datasets.all` - Dataset schemes (tfds, torch)

            **Web:**
                - `Scheme.web.all` - web related schemes (http, https, ftp, ftps)

        allow_query: If True, allows query parameters in URIs (e.g., ?key=value).
            Query parameters may contain sensitive data like API keys or tokens.
            Only enable for trusted use cases like signed URLs or parameterized APIs.
        allow_relative: If True, allows relative paths without schemes (for
            MLflow local artifacts).
        cloud_names: If True, validates bucket/container names against
            provider-specific rules (AWS S3, GCS, Azure).
        max_length: Maximum allowed URI length in characters. Defaults to 8192.
        require_host: If True, requires network location (host/bucket) for
            schemes that typically need it. Set to False for URNs, file://, or
            MLflow special schemes (runs:/, models:/).

    Returns:
        str: The validated URI with whitespace stripped.

    Raises:
        TypeError: If uri is not a string.
        ValueError: If uri is empty, exceeds max_length, has invalid format,
            unsupported scheme, missing required components, violates
            cloud provider naming rules, or contains query parameters when
            allow_query=False.

    Examples:
        >>> # Web (HTTPS)
        >>> validate_uri(
        ...     "https://example.com/path/resource?ref=homepage#section",
        ...     schemes=Scheme.web.all,
        ...     allow_query=True
        ... )
        'https://example.com/path/resource?ref=homepage#section'

        >>> # Web with signed query (allow_query=True)
        >>> validate_uri(
        ...     "https://cdn.example.com/file.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA...%2F20250101%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250101T000000Z&X-Amz-Expires=300&X-Amz-SignedHeaders=host&X-Amz-Signature=abcdef1234567890",
        ...     schemes=[Scheme.web.https],
        ...     allow_query=True
        ... )
        'https://cdn.example.com/file.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA...%2F20250101%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250101T000000Z&X-Amz-Expires=300&X-Amz-SignedHeaders=host&X-Amz-Signature=abcdef1234567890'

        >>> # PostgreSQL
        >>> validate_uri(
        ...     "postgresql://user:secret@db.example.com:5432/mydb?sslmode=require",
        ...     schemes=Scheme.db.sql.all,
        ...     allow_query=True
        ... )
        'postgresql://user:secret@db.example.com:5432/mydb?sslmode=require'

        >>> # BigQuery
        >>> validate_uri(
        ...     "bigquery://project-id/dataset/table",
        ...     schemes=Scheme.db.cloud.gcp.all
        ... )
        'bigquery://project-id/dataset/table'

        >>> # Redshift
        >>> validate_uri(
        ...     "redshift://cluster.region.redshift.amazonaws.com:5439/db",
        ...     schemes=Scheme.db.cloud.aws.all
        ... )
        'redshift://cluster.region.redshift.amazonaws.com:5439/db'

        >>> # MongoDB
        >>> validate_uri(
        ...     "mongodb://user:pass@localhost:27017/database",
        ...     schemes=Scheme.db.nosql.all
        ... )
        'mongodb://user:pass@localhost:27017/database'

        >>> # Redis
        >>> validate_uri(
        ...     "redis://cache.example.com:6379/0",
        ...     schemes=Scheme.db.nosql.all
        ... )
        'redis://cache.example.com:6379/0'

        >>> # Hugging Face Hub
        >>> validate_uri(
        ...     "hf://datasets/squad/train.parquet",
        ...     schemes=Scheme.ml.hub.all
        ... )
        'hf://datasets/squad/train.parquet'

        >>> # Combined: MLflow with BigQuery backend
        >>> validate_uri(
        ...     "bigquery://project/dataset/experiments",
        ...     schemes=(*Scheme.ml.all, *Scheme.db.all)
        ... )
        'bigquery://project/dataset/experiments'

        >>> # Local file path (file://)
        >>> validate_uri(
        ...     "file:///home/user/data.csv",
        ...     schemes=Scheme.local.all,
        ...     require_host=False
        ... )
        'file:///home/user/data.csv'

        >>> # Error: Unsupported scheme
        >>> validate_uri("unknown://example.com")
        Traceback (most recent call last):
        ...
        ValueError: unsupported URI scheme 'unknown'...
    """
    # Type validations
    if not isinstance(uri, str):
        raise TypeError(f"URI must be a string, got {fmt_type(uri)}")
    if not isinstance(schemes, (list, tuple, type(None))):
        raise TypeError(f"schemes must be a list or tuple, got {fmt_type(schemes)}")
    if not isinstance(max_length, int):
        raise TypeError(f"max_length must be a int, got {fmt_type(max_length)}")

    # Default allowed schemes for ML/DS context
    if schemes is None:
        schemes = (
            *Scheme.cloud(),
            *Scheme.bigdata(),
            *Scheme.ml.all,
            *Scheme.db.all,
            *Scheme.web.all,
            *Scheme.local.all,
        )

    # Strip whitespace
    uri = uri.strip()

    # Empty check
    if not uri:
        raise ValueError("URI cannot be empty")

    # Allow relative paths if specified (for MLflow local artifacts)
    if allow_relative and not "://" in uri and not uri.startswith("/"):
        # Relative path without scheme
        return uri

    # Length check
    if len(uri) > max_length:
        raise ValueError(
            f"URI exceeds maximum length of {max_length} characters, got {len(uri)}"
        )

    # Parse URI
    try:
        parsed = urlparse(uri)
    except Exception as e:
        raise ValueError(f"invalid URI format: {e}") from e

    # Validate scheme
    if not parsed.scheme:
        raise ValueError(f"invalid URI format: missing or invalid scheme")

    if parsed.scheme not in schemes:
        schemes_str = ", ".join(sorted(set(schemes)))
        raise ValueError(
            f"unsupported URI scheme '{parsed.scheme}'. Allowed schemes: {schemes_str}"
        )

    # Validate query parameters
    if not allow_query and parsed.query:
        raise ValueError(
            f"URI Query parameters are not allowed. Set allow_query=True to permit query strings"
        )

    # Scheme-specific validation (placeholders for future implementation)
    if parsed.scheme in AWSDatabaseSchemes.all:
        _validate_aws_db_uri(uri, parsed)
    elif parsed.scheme in AzureDatabaseSchemes.all:
        _validate_azure_db_uri(uri, parsed)
    elif parsed.scheme in GCPDatabaseSchemes.all:
        _validate_gcp_db_uri(uri, parsed)
    elif parsed.scheme == Scheme.ml.mlflow.models:
        _validate_mlflow_models_uri(uri, parsed)
    elif parsed.scheme == Scheme.ml.mlflow.runs:
        _validate_mlflow_runs_uri(uri, parsed)
    elif parsed.scheme == "neo4j" or parsed.scheme == "neo4js":
        _validate_neo4j_uri(uri, parsed)
    elif parsed.scheme in NoSQLSchemes.all and parsed.scheme.startswith("mongo"):
        _validate_mongodb_uri(uri, parsed)
    elif parsed.scheme in VectorSchemes.all:
        _validate_vector_db_uri(uri, parsed)

    # Schemes that don't require netloc
    no_netloc_schemes = {
        Scheme.db.sql.sqlite,
        Scheme.distributed.dbfs,
        Scheme.lakehouse.delta,
        Scheme.local.file,
        Scheme.local.urn,
        Scheme.ml.datasets.tfds,
        Scheme.ml.mlflow.models,
        Scheme.ml.mlflow.runs,
        Scheme.ml.tracking.mlflow,
        Scheme.ml.tracking.wandb,
    }

    # Validate network location (netloc) based on scheme
    if require_host and parsed.scheme not in no_netloc_schemes:
        if not parsed.netloc:
            raise ValueError(
                f"invalid URI format: missing network location (bucket/host)"
            )

    # Cloud storage specific validations (if enabled)
    if cloud_names:
        if parsed.scheme in AWSStorageSchemes.all:
            _validate_aws_s3_bucket(parsed.netloc)
        elif parsed.scheme in GCPStorageSchemes.all:
            _validate_gcp_gcs_bucket(parsed.netloc)
        elif parsed.scheme in AzureStorageSchemes.all:
            _validate_azure_storage(parsed.netloc, parsed.scheme)

    return uri


def _validate_aws_db_uri(uri: str, parsed) -> None:
    """Validate AWS database URI formats.

    Handles Redshift, DynamoDB, Athena, Timestream, and basics for RDS/Aurora/DocumentDB/Neptune.
    """
    scheme = parsed.scheme
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")
    parts = [p for p in path.split("/") if p]

    if scheme == AWSDatabaseSchemes.redshift:
        # redshift://host[:port]/database
        if not netloc:
            raise ValueError("Redshift URI must include cluster endpoint host")
        # Basic host validation
        if not re.match(r"^[A-Za-z0-9.-]+(:\d+)?$", netloc):
            raise ValueError(f"invalid Redshift host or port")
        if len(parts) < 1:
            raise ValueError(
                f"Redshift URI should specify database: redshift://host[:port]/<database>"
            )
        return

    if scheme == AWSDatabaseSchemes.dynamodb:
        # dynamodb://region/table
        if not netloc:
            raise ValueError(
                f"DynamoDB URI must include region as netloc: dynamodb://<region>/<table>"
            )
        if not re.match(r"^[a-z]{2}-[a-z]+-\d+$", netloc):
            # Be lenient, but nudge toward region pattern
            if "." in netloc:
                raise ValueError(
                    f"DynamoDB netloc should be a region identifier, not a host"
                )
        if len(parts) < 1:
            raise ValueError(
                f"DynamoDB URI must include table: dynamodb://region/<table>"
            )
        return

    if scheme == AWSDatabaseSchemes.athena:
        # athena://catalog/database[/table]
        if not netloc:
            raise ValueError(
                f"Athena URI must include catalog as netloc: athena://<catalog>/database[/table]"
            )
        if len(parts) < 1:
            raise ValueError(
                f"Athena URI must include database: athena://catalog/<database>[/table]"
            )
        return

    if scheme == AWSDatabaseSchemes.timestream:
        # timestream://region/database[/table]
        if not netloc:
            raise ValueError(
                f"Timestream URI must include region as netloc: timestream://<region>/database"
            )
        if len(parts) < 1:
            raise ValueError(
                f"Timestream URI must include database: timestream://region/<database>"
            )
        return

    if scheme in {
        AWSDatabaseSchemes.rds,
        AWSDatabaseSchemes.aurora,
        AWSDatabaseSchemes.documentdb,
        AWSDatabaseSchemes.neptune_db,
    }:
        # Require a host; path/database optional
        if not netloc:
            raise ValueError(f"{scheme} URI must include a host")
        if not re.match(r"^[A-Za-z0-9.-]+(:\d+)?$", netloc):
            raise ValueError(f"invalid host for {scheme}")
        return


def _validate_aws_s3_bucket(bucket: str) -> None:
    """Validate AWS S3 bucket naming rules."""
    if not bucket:
        return

    if len(bucket) < 3 or len(bucket) > 63:
        raise ValueError(f"invalid S3 bucket name '{bucket}': must be 3-63 characters")

    if bucket != bucket.lower():
        raise ValueError(f"invalid S3 bucket name '{bucket}': must be lowercase")

    if not re.match(r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$", bucket):
        raise ValueError(
            f"invalid S3 bucket name '{bucket}': must start/end with "
            "letter or number, and contain only lowercase letters, numbers, "
            "hyphens, and dots"
        )

    if ".." in bucket or ".-" in bucket or "-." in bucket:
        raise ValueError(
            f"invalid S3 bucket name '{bucket}': cannot contain consecutive "
            "dots or dot-dash combinations"
        )

    if re.match(r"^\d+\.\d+\.\d+\.\d+$", bucket):
        raise ValueError(
            f"invalid S3 bucket name '{bucket}': cannot be formatted as IP address"
        )


def _validate_azure_db_uri(uri: str, parsed) -> None:
    """Validate Azure database URI formats.

    Handles Cosmos DB, Synapse, SQL DW, and Azure SQL basic validation.
    """
    scheme = parsed.scheme
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")
    parts = [p for p in path.split("/") if p]

    if scheme == AzureDatabaseSchemes.cosmosdb:
        # cosmosdb://account-host[/database]
        if not netloc:
            raise ValueError(
                "Cosmos DB URI must include account host: cosmosdb://<account>.documents.azure.com[/db]"
            )
        # Accept either raw account or FQDN; be lenient
        account = netloc.split(".")[0]
        if not re.match(r"^[a-z0-9-]{3,44}$", account):
            raise ValueError("invalid Cosmos DB account name")
        return

    if scheme == AzureDatabaseSchemes.synapse or scheme == AzureDatabaseSchemes.sqldw:
        # synapse://workspace-host[/pool[/db]]
        if not netloc:
            raise ValueError("Synapse URI must include workspace/host")
        if not re.match(r"^[A-Za-z0-9.-]+(:\d+)?$", netloc):
            raise ValueError("invalid Synapse host")
        return

    if scheme == AzureDatabaseSchemes.azuresql:
        # azuresql://server.database.windows.net[/database]
        if not netloc:
            raise ValueError("Azure SQL URI must include server host")
        if not re.match(r"^[A-Za-z0-9.-]+(:\d+)?$", netloc):
            raise ValueError("invalid Azure SQL server host")
        return


def _validate_azure_storage(netloc: str, scheme: str) -> None:
    """Validate Azure storage account and container format."""
    if not netloc:
        return

    if scheme == Scheme.azure.adl:
        account_domain = netloc.split("/")[0]
        account = account_domain.split(".")[0]
        if not re.match(r"^[a-z0-9]{3,24}$", account):
            raise ValueError(
                f"invalid Azure Data Lake account '{account}': must be 3-24 "
                "characters, lowercase alphanumeric only"
            )
        return

    if scheme == Scheme.azure.az:
        container = netloc.split("/")[0]
        if not re.match(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$", container):
            raise ValueError(
                f"invalid Azure container name '{container}': must be 3-63 "
                "characters, lowercase alphanumeric and hyphens, start/end with "
                "letter or number"
            )
        return

    if "@" in netloc:
        container, account_domain = netloc.split("@", 1)

        if not re.match(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$", container):
            raise ValueError(
                f"invalid Azure container name '{container}': must be 3-63 "
                "characters, lowercase alphanumeric and hyphens"
            )

        account = account_domain.split(".")[0]
        if not re.match(r"^[a-z0-9]{3,24}$", account):
            raise ValueError(
                f"invalid Azure storage account '{account}': must be 3-24 "
                "characters, lowercase alphanumeric only"
            )


def _validate_gcp_db_uri(uri: str, parsed) -> None:
    """Validate GCP database URI formats.

    Handles BigQuery, Bigtable, Spanner, Firestore/Datastore specific validation.
    """
    scheme = parsed.scheme
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")
    parts = [p for p in path.split("/") if p]

    if scheme == GCPDatabaseSchemes.bigquery:
        # bigquery://project-id/dataset[/table]
        if not netloc:
            raise ValueError(
                "BigQuery URI must include project id as netloc: bigquery://<project-id>/dataset[/table]"
            )
        if len(parts) < 1:
            raise ValueError(
                "BigQuery URI must include at least a dataset: bigquery://project-id/<dataset>[/table]"
            )
        dataset = parts[0]
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", dataset):
            raise ValueError(f"invalid BigQuery dataset name '{dataset}'")
        if len(parts) >= 2:
            table = parts[1]
            # Table names can be quite flexible in BQ, be reasonably lenient
            if not re.match(r"^[A-Za-z0-9_$.]+$", table):
                raise ValueError(f"invalid BigQuery table name '{table}'")
        return

    if scheme == GCPDatabaseSchemes.bigtable:
        # bigtable://instance/table
        if not netloc:
            raise ValueError(
                "Bigtable URI must include instance as netloc: bigtable://<instance>/table"
            )
        if len(parts) < 1:
            raise ValueError(
                "Bigtable URI must include table: bigtable://instance/<table>"
            )
        return

    if scheme == GCPDatabaseSchemes.spanner:
        # spanner://instance/database
        if not netloc:
            raise ValueError(
                "Spanner URI must include instance as netloc: spanner://<instance>/database"
            )
        if len(parts) < 1:
            raise ValueError(
                "Spanner URI must include database: spanner://instance/<database>"
            )
        return

    if scheme in {GCPDatabaseSchemes.firestore, GCPDatabaseSchemes.datastore}:
        # firestore://project[/collection[/document]]
        if not netloc:
            raise ValueError(
                f"{scheme} URI must include project as netloc: {scheme}://<project>/..."
            )
        # No strict rules on path; if present, first part should be a collection name-like
        if parts:
            coll = parts[0]
            if not re.match(r"^[A-Za-z0-9_-]+$", coll):
                raise ValueError(f"invalid {scheme} collection '{coll}'")
        return


def _validate_gcp_gcs_bucket(bucket: str) -> None:
    """Validate Google Cloud Storage bucket naming rules."""
    if not bucket:
        return

    if len(bucket) < 3 or len(bucket) > 222:
        raise ValueError(
            f"invalid GCS bucket name '{bucket}': must be 3-63 characters "
            "(or up to 222 for domain-named buckets)"
        )

    if bucket != bucket.lower():
        raise ValueError(f"invalid GCS bucket name '{bucket}': must be lowercase")

    if not re.match(r"^[a-z0-9][a-z0-9._-]*[a-z0-9]$", bucket):
        raise ValueError(
            f"invalid GCS bucket name '{bucket}': must start/end with "
            "letter or number, and contain only lowercase letters, numbers, "
            "hyphens, underscores, and dots"
        )

    if re.match(r"^\d+\.\d+\.\d+\.\d+$", bucket):
        raise ValueError(
            f"invalid GCS bucket name '{bucket}': cannot be formatted as IP address"
        )


def _validate_mlflow_runs_uri(uri: str, parsed) -> None:
    """Validate MLflow runs:/ URI format.

    Args:
        uri: Original URI string.
        parsed: Parsed URI result.

    Raises:
        ValueError: If URI doesn't match MLflow runs:/ format.

    Note:
        Valid format: runs:/<run_id>/path/to/artifact
        Example: runs:/abc123def456/model/weights.pth
    """
    # Format: runs:/<run_id>/path
    # netloc will be empty, path should start with /
    if not parsed.path or not parsed.path.startswith("/"):
        raise ValueError(
            "invalid MLflow runs:/ URI format. Expected: runs:/<run_id>/path"
        )

    # Extract run_id (first path component after /)
    path_parts = parsed.path.lstrip("/").split("/", 1)
    if not path_parts or not path_parts[0]:
        raise ValueError(
            "invalid MLflow runs:/ URI format. Expected: runs:/<run_id>/path"
        )

    run_id = path_parts[0]
    # MLflow run IDs are typically 32-char hex strings
    if not re.match(r"^[a-f0-9]{32}$", run_id):
        # Also allow alphanumeric IDs (some backends use different formats)
        if not re.match(r"^[a-zA-Z0-9_-]+$", run_id):
            raise ValueError(
                f"invalid MLflow run ID '{run_id}'. Expected alphanumeric string"
            )


def _validate_mlflow_models_uri(uri: str, parsed) -> None:
    """Validate MLflow models:/ URI format.

    Args:
        uri: Original URI string.
        parsed: Parsed URI result.

    Raises:
        ValueError: If URI doesn't match MLflow models:/ format.

    Note:
        Valid formats:
        - models:/<name>/<version> (e.g., models:/my-model/1)
        - models:/<name>/<stage> (e.g., models:/my-model/Production)
    """
    # Format: models:/<name>/<version_or_stage>
    if not parsed.path or not parsed.path.startswith("/"):
        raise ValueError(
            "invalid MLflow models:/ URI format. Expected: models:/<name>/<version_or_stage>"
        )

    path_parts = parsed.path.lstrip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(
            "invalid MLflow models:/ URI format. Expected: models:/<name>/<version_or_stage>"
        )

    model_name, version_or_stage = path_parts[0], path_parts[1]

    if not model_name:
        raise ValueError("invalid MLflow models:/ URI: model name cannot be empty")

    if not version_or_stage:
        raise ValueError(
            "invalid MLflow models:/ URI: version or stage cannot be empty"
        )

    # Valid stages: None, Staging, Production, Archived
    valid_stages = {"None", "Staging", "Production", "Archived"}
    # Version should be numeric or a valid stage
    if not (version_or_stage.isdigit() or version_or_stage in valid_stages):
        raise ValueError(
            f"invalid MLflow model version or stage '{version_or_stage}'. "
            f"Expected: numeric version or one of {valid_stages}"
        )


def _validate_mongodb_uri(uri: str, parsed) -> None:
    """Validate MongoDB URI format.

    Handles connection string, replica sets, authentication.
    """
    # Accept both mongodb:// and mongo:// prefixes (the caller filters by scheme)
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if not netloc:
        raise ValueError("MongoDB URI must include host(s)")

    # Extract optional credentials and host list
    userinfo, hosts_part = (None, netloc)
    if "@" in netloc:
        userinfo, hosts_part = netloc.split("@", 1)
        # user[:password] may be empty user (discouraged), do a light check if provided
        if userinfo and ":" in userinfo:
            user, pwd = userinfo.split(":", 1)
            if not user:
                raise ValueError(
                    "MongoDB username cannot be empty when credentials are provided"
                )
        elif userinfo == "":
            raise ValueError(
                "MongoDB credentials marker '@' present but empty userinfo"
            )

    # Validate one or more hosts separated by commas
    host_re = re.compile(r"^[A-Za-z0-9._-]+(?::\d{1,5})?$")
    for host in [h for h in hosts_part.split(",") if h]:
        if not host_re.match(host):
            raise ValueError(f"invalid MongoDB host entry '{host}'")

    # Optional database in path: mongodb://host[:port]/database
    if path:
        # Database names cannot contain "/\\ . \" $" and must be <= 63 bytes; do a light regex
        if not re.match(r'^[^/\\\."$]{1,63}$', path):
            raise ValueError(f"invalid MongoDB database name '{path}'")

    # Query parameters like replicaSet, authSource are allowed but not validated here
    return


def _validate_neo4j_uri(uri: str, parsed) -> None:
    """Validate Neo4j URI format.

    Handles bolt/neo4j protocols, routing, and database names.
    """
    scheme = parsed.scheme
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    # Basic host check
    if not netloc:
        raise ValueError("Neo4j URI must include host")
    if not re.match(r"^[A-Za-z0-9.-]+(?::\d+)?$", netloc):
        raise ValueError("invalid Neo4j host or port")

    # Optional path can specify database like /db/<name>
    if path:
        parts = [p for p in path.split("/") if p]
        if parts:
            # Accept either 'db/<name>' or a single database name
            if parts[0] == "db":
                if len(parts) < 2:
                    raise ValueError(
                        "Neo4j URI path 'db' must be followed by database name"
                    )
                db = parts[1]
            else:
                db = parts[0]
            if not re.match(r"^[A-Za-z0-9_-]+$", db):
                raise ValueError(f"invalid Neo4j database name '{db}'")
    return


def _validate_vector_db_uri(uri: str, parsed) -> None:
    """Validate vector database URI formats.

    Handles Pinecone, Weaviate, Qdrant, Milvus, Chroma specific validation.
    """
    scheme = parsed.scheme
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    # For all vector DBs, require a host (netloc)
    if not netloc:
        raise ValueError(f"{scheme} URI must include host")

    host_ok = re.match(r"^[A-Za-z0-9.-]+(?::\d+)?$", netloc) is not None
    if not host_ok and "," in netloc:
        # Some self-hosted deployments might include multiple nodes separated by commas
        # Validate each host:port segment
        parts = [h for h in netloc.split(",") if h]
        host_ok = all(re.match(r"^[A-Za-z0-9.-]+(?::\d+)?$", p) for p in parts)
    if not host_ok:
        raise ValueError(f"invalid host format for {scheme} URI")

    if scheme == VectorSchemes.pinecone:
        # Typical: pinecone://index-xxx.svc.[env].pinecone.io
        # Be lenient: ensure contains 'pinecone' domain hint if FQDN-like
        if "." in netloc and "pinecone" not in netloc:
            raise ValueError(
                "Pinecone host should contain 'pinecone' domain when using FQDN"
            )
        return

    if scheme == VectorSchemes.weaviate:
        # weaviate://host[:port][/class]
        return

    if scheme == VectorSchemes.qdrant:
        # qdrant://host[:port][/collections/<name>]
        if path and not re.match(r"^(collections/)?[A-Za-z0-9_.-]+(/.*)?$", path):
            raise ValueError(
                "invalid Qdrant path; expected 'collections/<name>' or empty"
            )
        return

    if scheme == VectorSchemes.milvus:
        # milvus://host[:port]
        return

    if scheme in {VectorSchemes.chroma, VectorSchemes.chromadb}:
        # chroma://host[:port]
        return
