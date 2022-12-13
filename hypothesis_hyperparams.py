
### This script has the following purposes:
## 1. It contains the hypotheses for NLI
## 2. It contains the format_text function for each dataset.
# for each dataset, texts need to be formatted differently depending on the hypotheses, whether context sentences are available
# and other properties of datasets like column names

# import relevant packages
from collections import OrderedDict
import pandas as pd
import numpy as np

np.random.seed(42)


def hypothesis_hyperparams(dataset_name=None, df_cl=None, embeddings=False):
    # hypotheses and text templates
    # determines different hypotheses format (for NLI) and different text formats (for all classifiers) - both are treated as hyperparameters during training

    ### sentiment-news-econ
    if dataset_name == "sentiment-news-econ":
        hypothesis_hyperparams_dic = OrderedDict({
            "template_quote":
                {"positive": "The quote is overall positive",
                 "negative": "The quote is overall negative",
                 },
            "template_complex":  # ! performed best for most train sizes
                {"positive": "The economy is performing well overall",
                 "negative": "The economy is performing badly overall",
                 },
            "template_not_nli":
                {"placeholder": "placeholder"},
        })
        ## sort hypotheses alphabetically by label text
        # hypothesis_hyperparams_dic = OrderedDict()
        for key, value in hypothesis_hyperparams_dic.items():
            hypothesis_hyperparams_dic.update({key: dict(sorted(value.items(), key=lambda x: x[0]))})
        print(hypothesis_hyperparams_dic)
        print(dataset_name)

        ### text formatting function
        def format_text(df=None, text_format=None, embeddings=embeddings):
            if text_format == 'template_not_nli':
                df["text_prepared"] = df.text
            elif text_format == 'template_quote':
                df["text_prepared"] = 'The quote: "' + df.text + '" - end of the quote.'
            elif text_format == 'template_complex':
                df["text_prepared"] = df.text
            else:
                raise Exception(f'Hypothesis template not found for: {text_format}')
            return df.copy(deep=True)



    ### coronanet
    if dataset_name == "coronanet":
        ## Explicit labels
        # shorter hypotheses
        explicit_labels_dic_short = OrderedDict({
            'Anti-Disinformation Measures': "measures against disinformation",
            'COVID-19 Vaccines': "COVID-19 vaccines",
            'Closure and Regulation of Schools': "regulating schools",
            'Curfew': "a curfew",
            'Declaration of Emergency': "declaration of emergency",
            'External Border Restrictions': "external border restrictions",
            'Health Monitoring': "health monitoring",
            'Health Resources': "health resources, materials, infrastructure, personnel, mask purchases",
            'Health Testing': "health testing",
            'Hygiene': "hygiene",
            'Internal Border Restrictions': "internal border restrictions",
            'Lockdown': "a lockdown",
            'New Task Force, Bureau or Administrative Configuration': "a new administrative body",
            'Public Awareness Measures': "public awareness measures",
            'Quarantine': "quarantine",
            'Restriction and Regulation of Businesses': "restricting or regulating businesses",
            'Restriction and Regulation of Government Services': "restricting or regulating government services or public facilities",
            'Restrictions of Mass Gatherings': "restrictions of mass gatherings",
            'Social Distancing': "social distancing, reducing contact, mask wearing",
            "Other Policy Not Listed Above": "something other than regulation of businesses, government, gatherings, distancing, quarantine, lockdown, curfew, emergency, vaccine, disinformation, schools, borders or travel, testing, resources. It is not about any of these topics."
        })
        #https://www.coronanet-project.org/assets/CoronaNet_Codebook.pdf
        explicit_labels_dic_long = OrderedDict({
            'Anti-Disinformation Measures': "measures against disinformation: Efforts by the government to limit the spread of false, inaccurate or harmful information",
            'COVID-19 Vaccines': "COVID-19 vaccines. A policy regarding the research and development, or regulation, or production, or purchase, or distribution of a vaccine.",
            'Closure and Regulation of Schools': "regulating schools and educational establishments. For example closing an educational institution, or allowing educational institutions to open with or without certain conditions.",
            'Curfew': "a curfew: Domestic freedom of movement is limited during certain times of the day",
            'Declaration of Emergency': "declaration of a state of national emergency",
            'External Border Restrictions': "external border restrictions: The ability to enter or exit country borders is reduced.",
            'Health Monitoring': "health monitoring of individuals who are likely to be infected.",
            'Health Resources': "health resources: For example medical equipment, number of hospitals, health infrastructure, personnel (e.g. doctors, nurses), mask purchases",
            'Health Testing': "health testing of large populations regardless of their likelihood of being infected.",
            'Hygiene': "hygiene: Promotion of hygiene in public spaces, for example disinfection in subways or burials.",
            'Internal Border Restrictions': "internal border restrictions: The ability to move freely within the borders of a country is reduced.",
            'Lockdown': "a lockdown: People are obliged shelter in place and are only allowed to leave their shelter for specific reasons",
            'New Task Force, Bureau or Administrative Configuration': "a new administrative body, for example a new task force, bureau or administrative configuration.",
            'Public Awareness Measures': "public awareness measures or efforts to disseminate or gather reliable information, for example information on health prevention.",
            'Quarantine': "quarantine. People are obliged to isolate themselves if they are infected.",
            'Restriction and Regulation of Businesses': "restricting or regulating businesses, private commercial activities: For example closing down commercial establishments, or allowing commercial establishments to open with or without certain conditions.",
            'Restriction and Regulation of Government Services': "restricting or regulating government services or public facilities: For example closing down government services, or allowing government services to operate with or without certain conditions.",
            'Restrictions of Mass Gatherings': "restrictions of mass gatherings: The number of people allowed to congregate in a place is limited",
            'Social Distancing': "social distancing, reducing contact between individuals in public spaces, mask wearing.",
            "Other Policy Not Listed Above": "something other than regulation of businesses, government, gatherings, distancing, quarantine, lockdown, curfew, emergency, vaccines, disinformation, schools, borders or travel, testing, health resources. It is not about any of these topics."
        })
        ## double check that explicit label map keys correspond to dataset label_text
        label_hypo_keys_all = [key for key in explicit_labels_dic_short]
        labels_all = df_cl.label_text.unique().tolist()
        assert all(elem in label_hypo_keys_all for elem in labels_all)
        assert all(elem in labels_all for elem in label_hypo_keys_all)

        ### Hypothesis template
        hypothesis_templates_dic = {
            #"template_simple": "It is about {}.",
            "template_quote": "The quote is about {}.",
            #"template_policy": "The policy is about {}.",
            "template_not_nli": "NA",
        }

        ## merge hypotheses template with explicit labels
        hypo_short_dic_dic = OrderedDict()
        for key_hypo, value_hypo in hypothesis_templates_dic.items():
            hypo_short_dic = OrderedDict()
            for key_label, value_label in explicit_labels_dic_short.items():
                hypo_short_dic.update({key_label: value_hypo.format(value_label)})
            hypo_short_dic_dic.update({key_hypo: hypo_short_dic})
            if "not_nli" not in key_hypo:
                hypo_long_dic = OrderedDict()
                for key_label, value_label in explicit_labels_dic_long.items():
                    hypo_long_dic.update({key_label: value_hypo.format(value_label)})
                hypo_short_dic_dic.update({f"{key_hypo}_long_hypo": hypo_long_dic})

        ## sort hypotheses alphabetically by label text
        hypothesis_hyperparams_dic = OrderedDict()
        for key, value in hypo_short_dic_dic.items():
            hypothesis_hyperparams_dic.update({key: dict(sorted(value.items(), key=lambda x: x[0].lower()))})
        print(hypothesis_hyperparams_dic)
        print(dataset_name)

        ### text formatting function
        def format_text(df=None, text_format=None, embeddings=embeddings):
            if (text_format == 'template_not_nli') or (text_format == 'template_simple'):
                df["text_prepared"] = df.text
            elif (text_format == 'template_quote') or (text_format == 'template_quote_long_hypo'):
                df["text_prepared"] = 'The quote: "' + df.text + '".'
            #elif text_format == 'template_policy':
            #    df["text_prepared"] = 'The policy: "' + df.text + '".'
            else:
                raise Exception(f'Hypothesis template not found for: {text_format}')
            return df.copy(deep=True)


    ### cap-us-court
    if dataset_name == "cap-us-court":
        ## Explicit labels
        # short explicit labels
        explicit_labels_dic_short = OrderedDict({
            'Agriculture': "agriculture",
            #'Culture': "cultural policy",
            'Civil Rights': "civil rights, or minorities, or civil liberties",
            'Defense': "defense, or military",
            'Domestic Commerce': "banking, or finance, or commerce",
            'Education': "education",
            'Energy': "energy, or electricity, or fossil fuels",
            'Environment': "the environment, or water, or waste, or pollution",
            'Foreign Trade': "foreign trade",
            'Government Operations': "government operations, or administration",
            'Health': "health",
            'Housing': "community development, or housing issues",
            'Immigration': "migration",
            'International Affairs': "international affairs, or foreign aid",
            'Labor': "employment, or labour",
            'Law and Crime': "law, crime, or family issues",
            'Macroeconomics': "macroeconomics",
            # 'Other': "other, miscellaneous",
            'Public Lands': "public lands, or water management",
            'Social Welfare': "social welfare",
            'Technology': "space, or science, or technology, or communications",
            'Transportation': "transportation",
        })
        explicit_labels_dic_long = OrderedDict({
            'Agriculture': "agriculture, for example: agricultural foreign trade, or subsidies to farmers, or food inspection and safety, or agricultural marketing, or animal and crop disease, or fisheries, or R&D",
            # 'Culture': "cultural policy",
            'Civil Rights': "civil rights, for example: minority/gender/age/handicap discrimination, or voting rights, or freedom of speech, or privacy",
            'Defense': "defense, for example: defense alliances, or military intelligence, or military readiness, or nuclear arms, or military aid, or military personnel issues, or military procurement, or reserve forces, or hazardous waste, or civil defense and terrorism, or contractors, or foreign operations, or R&D",
            'Domestic Commerce': "domestic commerce, for example: banking, or securities and commodities, or consumer finance, or insurance regulation, or bankruptcy, or corporate management, or small businesses, or copyrights and patents, or disaster relief, or tourism, or consumer safety, or sports regulation, or R&D",
            'Education': "education, for example: higher education, or education finance, or schools, or education of underprivileged, or vocational education, or education for handicapped, or excellence, or R&D",
            'Energy': "energy, for example: nuclear energy and safety, or electricity, or natural gas & oil, or coal, or alternative and renewable energy, or conservation, or R&D",
            'Environment': "the environment, for example: drinking water, or waste disposal, or hazardous waste, or air pollution, or recycling, or species and forest protection, or conservation, or R&D",
            'Foreign Trade': "foreign trade, for example: trade agreements, or exports, or private investments, or competitiveness, or tariff and imports, or exchange rates",
            'Government Operations': "government operations, for example: intergovernmental relations, or agencies, or bureaucracy, or postal service, or civil employees, or appointments, or national currency, or government procurement, or government property management, or tax administration, or public scandals, or government branch relations, or political campaigns, or census, or capital city, or national holidays",
            'Health': "health, for example: health care reform, or health insurance, or drug industry, or medical facilities, or disease prevention, or infants and children, or mental health, or drug/alcohol/tobacco abuse, or R&D",
            'Housing': "housing, for example: community development, or urban development, or rural housing, low-income assistance for housing, housing for veterans/elderly/homeless, or R&D",
            'Immigration': "migration, for example: immigration, or refugees, or citizenship",
            'International Affairs': "international affairs, for example: foreign aid, or international resources exploitation, or developing countries, or international finance, or western Europe, or specific countries, or human rights, or international organisations, or international terrorism, or diplomats",
            'Labor': "labour, for example: worker safety, or employment training, or employee benefits, or labor unions, or fair labor standards, or youth employment, or migrant and seasonal workers",
            'Law and Crime': "law and crime, for example: law enforcement agencies, or white collar crime, or illegal drugs, or court administration, or prisons, or juvenile crime, or child abuse, or family issues, or criminal and civil code, or police",
            'Macroeconomics': "macroeconomics, for example: interest rates, or unemployment, or monetary policy, or national budget, or taxes, or industrial policy",
            # 'Other': "other things, miscellaneous",
            'Public Lands': "public lands, for example: national parks, or indigenous affairs, or public lands, or water resources, or dependencies and territories",
            'Social Welfare': "social welfare, for example: low-income assistance, or elderly assistance, or disabled assistance, or volunteer associations, or child care, or social welfare",
            'Technology': "technology, for example: government space programs, or commercial use of space, or science transfer, or telecommunications, or regulation of media, or weather science, or computers, or internet, or R&D",
            'Transportation': "transportation, for example: mass transportation, or highways, or air travel, or railroads, or maritime, or infrastructure, or R&D",
        })

        ## double check that explicit label map keys correspond to dataset label_text
        label_hypo_keys_all = [key for key in explicit_labels_dic_short]
        labels_all = df_cl.label_text.unique().tolist()
        assert all(elem in label_hypo_keys_all for elem in labels_all)
        assert all(elem in labels_all for elem in label_hypo_keys_all)
        # [print(label) for label in label_hypo_keys_all if label not in labels_all]

        ### Hypothesis template
        hypothesis_templates_dic = {
            # "template_simple": "It is about {}.",
            "template_quote": "The quote is about {}.",
            # "template_complex": "The court case is about {}.",  # seems to have worse/much more volatile performance
            "template_not_nli": "NA",
        }

        ## merge hypotheses template with explicit labels
        hypo_short_dic_dic = OrderedDict()
        for key_hypo, value_hypo in hypothesis_templates_dic.items():
            hypo_short_dic = OrderedDict()
            for key_label, value_label in explicit_labels_dic_short.items():
                hypo_short_dic.update({key_label: value_hypo.format(value_label)})
            hypo_short_dic_dic.update({key_hypo: hypo_short_dic})
            if "not_nli" not in key_hypo:
                hypo_long_dic = OrderedDict()
                for key_label, value_label in explicit_labels_dic_long.items():
                    hypo_long_dic.update({key_label: value_hypo.format(value_label)})
                hypo_short_dic_dic.update({f"{key_hypo}_long_hypo": hypo_long_dic})

        ## sort hypotheses alphabetically by label text
        hypothesis_hyperparams_dic = OrderedDict()
        for key, value in hypo_short_dic_dic.items():
            hypothesis_hyperparams_dic.update({key: dict(sorted(value.items(), key=lambda x: x[0].lower()))})
        print(hypothesis_hyperparams_dic)
        print(dataset_name)

        ### text formatting function
        def format_text(df=None, text_format=None, embeddings=embeddings):
            if (text_format == 'template_not_nli') or (text_format == 'template_simple'):
                df["text_prepared"] = df.text
            elif (text_format == 'template_quote') or (text_format == 'template_quote_long_hypo'):
                df["text_prepared"] = 'The quote: "' + df.text + '" - end of the quote.'
            elif (text_format == 'template_complex') or (text_format == 'template_complex_long_hypo'):
                df["text_prepared"] = 'The court case: "' + df.text + '" - end of the court case.'
            else:
                raise Exception(f'Hypothesis template not found for: {text_format}')
            return df.copy(deep=True)



    ### cap-sotu
    if dataset_name == "cap-sotu":
        ## Explicit labels
        # short explicit labels
        # ! can probably be improved
        explicit_labels_dic_short = OrderedDict({
            'Agriculture': "agriculture",
            'Culture': "cultural policy",
            'Civil Rights': "civil rights, or minorities, or civil liberties",
            'Defense': "defense, or military",
            'Domestic Commerce': "banking, or finance, or commerce",
            'Education': "education",
            'Energy': "energy, or electricity, or fossil fuels",
            'Environment': "the environment, or water, or waste, or pollution",
            'Foreign Trade': "foreign trade",
            'Government Operations': "government operations, or administration",
            'Health': "health",
            'Housing': "community development, or housing issues",
            'Immigration': "migration",
            'International Affairs': "international affairs, or foreign aid",
            'Labor': "employment, or labour",
            'Law and Crime': "law, crime, or family issues",
            'Macroeconomics': "macroeconomics",
            'Other': "other, miscellaneous",
            'Public Lands': "public lands, or water management",
            'Social Welfare': "social welfare",
            'Technology': "space, or science, or technology, or communications",
            'Transportation': "transportation",
        })
        ## long hypos
        explicit_labels_dic_long = OrderedDict({
            'Agriculture': "agriculture, for example: agricultural foreign trade, or subsidies to farmers, or food inspection and safety, or agricultural marketing, or animal and crop disease, or fisheries, or R&D",
            'Culture': "cultural policy",
            'Civil Rights': "civil rights, for example: minority/gender/age/handicap discrimination, or voting rights, or freedom of speech, or privacy",
            'Defense': "defense, for example: defense alliances, or military intelligence, or military readiness, or nuclear arms, or military aid, or military personnel issues, or military procurement, or reserve forces, or hazardous waste, or civil defense and terrorism, or contractors, or foreign operations, or R&D",
            'Domestic Commerce': "domestic commerce, for example: banking, or securities and commodities, or consumer finance, or insurance regulation, or bankruptcy, or corporate management, or small businesses, or copyrights and patents, or disaster relief, or tourism, or consumer safety, or sports regulation, or R&D",
            'Education': "education, for example: higher education, or education finance, or schools, or education of underprivileged, or vocational education, or education for handicapped, or excellence, or R&D",
            'Energy': "energy, for example: nuclear energy and safety, or electricity, or natural gas & oil, or coal, or alternative and renewable energy, or conservation, or R&D",
            'Environment': "the environment, for example: drinking water, or waste disposal, or hazardous waste, or air pollution, or recycling, or species and forest protection, or conservation, or R&D",
            'Foreign Trade': "foreign trade, for example: trade agreements, or exports, or private investments, or competitiveness, or tariff and imports, or exchange rates",
            'Government Operations': "government operations, for example: intergovernmental relations, or agencies, or bureaucracy, or postal service, or civil employees, or appointments, or national currency, or government procurement, or government property management, or tax administration, or public scandals, or government branch relations, or political campaigns, or census, or capital city, or national holidays",
            'Health': "health, for example: health care reform, or health insurance, or drug industry, or medical facilities, or disease prevention, or infants and children, or mental health, or drug/alcohol/tobacco abuse, or R&D",
            'Housing': "housing, for example: community development, or urban development, or rural housing, low-income assistance for housing, housing for veterans/elderly/homeless, or R&D",
            'Immigration': "migration, for example: immigration, or refugees, or citizenship",
            'International Affairs': "international affairs, for example: foreign aid, or international resources exploitation, or developing countries, or international finance, or western Europe, or specific countries, or human rights, or international organisations, or international terrorism, or diplomats",
            'Labor': "labour, for example: worker safety, or employment training, or employee benefits, or labor unions, or fair labor standards, or youth employment, or migrant and seasonal workers",
            'Law and Crime': "law and crime, for example: law enforcement agencies, or white collar crime, or illegal drugs, or court administration, or prisons, or juvenile crime, or child abuse, or family issues, or criminal and civil code, or police",
            'Macroeconomics': "macroeconomics, for example: interest rates, or unemployment, or monetary policy, or national budget, or taxes, or industrial policy",
            'Other': "other things, miscellaneous",
            'Public Lands': "public lands, for example: national parks, or indigenous affairs, or public lands, or water resources, or dependencies and territories",
            'Social Welfare': "social welfare, for example: low-income assistance, or elderly assistance, or disabled assistance, or volunteer associations, or child care, or social welfare",
            'Technology': "technology, for example: government space programs, or commercial use of space, or science transfer, or telecommunications, or regulation of media, or weather science, or computers, or internet, or R&D",
            'Transportation': "transportation, for example: mass transportation, or highways, or air travel, or railroads, or maritime, or infrastructure, or R&D",
        })

        ## double check that explicit label map keys correspond to dataset label_text
        label_hypo_keys_all = [key for key in explicit_labels_dic_short]
        labels_all = df_cl.label_text.unique().tolist()
        assert all(elem in label_hypo_keys_all for elem in labels_all)
        assert all(elem in labels_all for elem in label_hypo_keys_all)

        ## Hypothesis template
        hypothesis_templates_dic = {
            # "template_simple": "It is about {}",
            "template_quote": "The quote is about {}.",
            "template_quote_context": "The quote is about {}.",
            "template_not_nli": "NA",
            "template_not_nli_context": "NA",
        }
        ## merge hypotheses template with explicit labels
        hypo_short_dic_dic = OrderedDict()
        for key_hypo, value_hypo in hypothesis_templates_dic.items():
            hypo_short_dic = OrderedDict()
            for key_label, value_label in explicit_labels_dic_short.items():
                hypo_short_dic.update({key_label: value_hypo.format(value_label)})
            hypo_short_dic_dic.update({key_hypo: hypo_short_dic})
            # add hypo templates for long hypotheses
            if "not_nli" not in key_hypo:
                hypo_long_dic = OrderedDict()
                for key_label, value_label in explicit_labels_dic_long.items():
                    hypo_long_dic.update({key_label: value_hypo.format(value_label)})
                hypo_short_dic_dic.update({f"{key_hypo}_long_hypo": hypo_long_dic})

        ## sort hypotheses alphabetically by label text
        hypothesis_hyperparams_dic = OrderedDict()
        for key, value in hypo_short_dic_dic.items():
            hypothesis_hyperparams_dic.update({key: dict(sorted(value.items(), key=lambda x: x[0].lower()))})
        print(hypothesis_hyperparams_dic)
        print(dataset_name)

        ### text formatting function
        def format_text(df=None, text_format=None, embeddings=embeddings):
            if (text_format == 'template_not_nli') or (text_format == 'template_simple'):
                df["text_prepared"] = df.text_original
            elif (text_format == 'template_not_nli_context') and (embeddings == False):
                df["text_prepared"] = df.text_preceding.fillna("") + " " + df.text_original.fillna("") + " " + df.text_following.fillna("")
            elif (text_format == 'template_not_nli_context') and (embeddings == True):
                df["text_prepared"] = df.apply(lambda x: np.mean([x["text_original"], x["text_original"], x["text_original"], x["text_original"], x["text_preceding"], x["text_following"]], dtype=object, axis=0), axis=1)  # weigh target text x times higher
            elif (text_format == 'template_quote') or (text_format == 'template_quote_long_hypo'):
                df["text_prepared"] = 'The quote: "' + df.text_original + '" - end of the quote.'
            elif (text_format == 'template_quote_context') or (text_format == 'template_quote_context_long_hypo'):
                df["text_prepared"] = df.text_preceding.fillna("") + '. The quote: "' + df.text_original.fillna("") + '" - end of the quote. ' + df.text_following.fillna("")
            else:
                raise Exception(f'Hypothesis template not found for: {text_format}')
            return df.copy(deep=True)


    ### manifesto-8
    if dataset_name == "manifesto-8":
        ### domain hypotheses
        ## Explicit labels
        # short explicit labels
        explicit_labels_dic_short = OrderedDict({
            "Economy": "economy, or technology, or infrastructure, or free market",
            "External Relations": "international relations, or foreign policy, or military",
            "Fabric of Society": "law and order, or multiculturalism, or national way of life, or traditional morality",
            "Freedom and Democracy": "democracy, or freedom, or human rights, or constitutionalism",
            "Political System": "governmental efficiency, or political authority, or decentralisation, or corruption",
            "Social Groups": "agriculture, or social groups, or labour groups, or minorities",
            "Welfare and Quality of Life": "welfare, or education, or environment, or equality, or culture",
            "No other category applies": "something other than the topics economy, international relations, society, freedom and democracy, political system, social groups, welfare. It is about non of these topics"
        })
        # long explicit labels
        hypo_label_dic_long = OrderedDict({
            "Economy": "economy, free market economy, incentives, market regulation, economic planning, cooperation of government, employers and unions, protectionism, economic growth, technology and infrastructure, nationalisation, neoliberalism, marxism, sustainability",
            "External Relations": "international relations, foreign policy, anti-imperialism, military, peace, internationalism, European Union",
            "Fabric of Society": "society, national way of life, immigration, traditional morality, law and order, civic mindedness, solidarity, multiculturalism, diversity",
            "Freedom and Democracy": "democracy, freedom, human rights, constitutionalism, representative or direct democracy",
            "Political System": "political system, centralisation, governmental and administrative efficiency, political corruption, political authority",
            "Social Groups": "social groups, labour groups, agriculture and farmers, middle class and professional groups, minority groups, women, students, old people",
            "Welfare and Quality of Life": "welfare and quality of life, environmental protection, culture, equality, welfare state, education",
            "No other category applies": "something other than the topics economy, international relations, society, freedom and democracy, political system, social groups, welfare. It is about non of these topics"
        })

        ## double check that explicit label map keys correspond to dataset label_text
        label_hypo_keys_all = [key for key in explicit_labels_dic_short]
        labels_all = df_cl.label_text.unique().tolist()
        assert all(elem in label_hypo_keys_all for elem in labels_all)
        assert all(elem in labels_all for elem in label_hypo_keys_all)

        ## Hypothesis template
        hypothesis_templates_dic = {
            # "template_simple": "It is about {}",
            "template_quote": "The quote is about {}.",
            "template_quote_context": "The quote is about {}.",
            "template_not_nli": "NA",
            "template_not_nli_context": "NA",
        }
        ## merge hypotheses template with explicit labels
        hypo_short_dic_dic = OrderedDict()
        for key_hypo, value_hypo in hypothesis_templates_dic.items():
            hypo_short_dic = OrderedDict()
            for key_label, value_label in explicit_labels_dic_short.items():
                hypo_short_dic.update({key_label: value_hypo.format(value_label)})
            hypo_short_dic_dic.update({key_hypo: hypo_short_dic})
            # add hypo templates for long hypotheses
            if "not_nli" not in key_hypo:
                hypo_long_dic = OrderedDict()
                for key_label, value_label in hypo_label_dic_long.items():
                    hypo_long_dic.update({key_label: value_hypo.format(value_label)})
                hypo_short_dic_dic.update({f"{key_hypo}_long_hypo": hypo_long_dic})

        ## sort hypotheses alphabetically by label text
        hypothesis_hyperparams_dic = OrderedDict()
        for key, value in hypo_short_dic_dic.items():
            hypothesis_hyperparams_dic.update({key: dict(sorted(value.items(), key=lambda x: x[0].lower()))})
        print(hypothesis_hyperparams_dic)
        print(dataset_name)

        ### text formatting function
        def format_text(df=None, text_format=None, embeddings=embeddings):
            if (text_format == 'template_not_nli') or (text_format == 'template_simple'):
                df["text_prepared"] = df.text_original
            elif (text_format == 'template_not_nli_context') and (embeddings == False):
                df["text_prepared"] = df.text_preceding.fillna("") + df.text_original.fillna("") + df.text_following.fillna("")
            elif (text_format == 'template_not_nli_context') and (embeddings == True):
                df["text_prepared"] = df.apply(lambda x: np.mean([x["text_original"], x["text_original"], x["text_original"], x["text_original"], x["text_preceding"], x["text_following"]], dtype=object, axis=0), axis=1)  # weigh target text x times higher
            elif (text_format == 'template_quote') or (text_format == 'template_quote_long_hypo'):
                df["text_prepared"] = 'The quote: "' + df.text_original + '" - end of the quote.'
            elif (text_format == 'template_quote_context') or (text_format == 'template_quote_context_long_hypo'):
                df["text_prepared"] = df.text_preceding.fillna("") + '. The quote: "' + df.text_original.fillna("") + '" - end of the quote. ' + df.text_following.fillna("")
            else:
                raise Exception(f'Hypothesis template not found for: {text_format}')
            return df.copy(deep=True)


    ### manifesto-military
    if dataset_name == "manifesto-military":
        hypothesis_hyperparams_dic = OrderedDict({
            "template_quote":
                {"Military: Positive": "The quote is positive towards the military",
                 "Military: Negative": "The quote is negative towards the military",
                 "Other": "The quote is not about military or defense"
                 },
            "template_quote_2":  # ! performed best for most train sizes
                {"Military: Positive": "The quote is positive towards the military, for example for military spending, defense, military treaty obligations.",
                 "Military: Negative": "The quote is negative towards the military, for example against military spending, for disarmament, against conscription.",
                 "Other": "The quote is not about military or defense"
                 },
            "template_quote_context":
                {"Military: Positive": "The quote is positive towards the military",
                 "Military: Negative": "The quote is negative towards the military",
                 "Other": "The quote is not about military or defense"
                 },
            "template_quote_2_context":  # ! performed best for most train sizes
                {"Military: Positive": "The quote is positive towards the military, for example for military spending, defense, military treaty obligations.",
                 "Military: Negative": "The quote is negative towards the military, for example against military spending, for disarmament, against conscription.",
                 "Other": "The quote is not about military or defense"
                 },
            "template_not_nli":
                {"placeholder": "placeholder"},
            "template_not_nli_context":
                {"placeholder": "placeholder"},
        })
        ## sort hypotheses alphabetically by label text
        # hypothesis_hyperparams_dic = OrderedDict()
        for key, value in hypothesis_hyperparams_dic.items():
            hypothesis_hyperparams_dic.update({key: dict(sorted(value.items(), key=lambda x: x[0]))})  # .lower()
        print(hypothesis_hyperparams_dic)
        print(dataset_name)

        ### text formatting function
        def format_text(df=None, text_format=None, embeddings=embeddings):
            if text_format == 'template_not_nli':
                df["text_prepared"] = df.text_original
            elif (text_format == 'template_not_nli_context') and (embeddings == False):
                df["text_prepared"] = df.text_preceding.fillna("") + " " + df.text_original.fillna("") + " " + df.text_following.fillna("")
            elif (text_format == 'template_not_nli_context') and (embeddings == True):  # for classical_ml with embedding input, including embeddings of context sentences
                df["text_prepared"] = df.apply(lambda x: np.mean([x["text_original"], x["text_original"], x["text_original"], x["text_original"], x["text_preceding"], x["text_following"]], dtype=object, axis=0), axis=1)  # weigh target text x times higher
            elif (text_format == 'template_quote') or (text_format == 'template_quote_2'):
                df["text_prepared"] = 'The quote: "' + df.text_original.fillna("") + '" - end of the quote.'
            elif (text_format == 'template_quote_context') or (text_format == 'template_quote_2_context'):
                df["text_prepared"] = df.text_preceding.fillna("") + '. The quote: "' + df.text_original.fillna("") + '" - end of the quote. ' + df.text_following.fillna("")
            # elif text_format == 'template_complex':
            #  df["text_prepared"] = df.text_preceding.fillna("") + df.text_original.fillna("") + df.text_following.fillna("")
            else:
                raise Exception(f'Hypothesis template not found for: {text_format}')
            return df.copy(deep=True)



    ### manifesto-protectionism
    if dataset_name == "manifesto-protectionism":
        hypothesis_hyperparams_dic = OrderedDict({
            # "template_quote":   # didn't perform as well
            #    {"Protectionism: Positive": "The quote is positive towards protectionism",
            #     "Protectionism: Negative": "The quote is negative towards protectionism",
            #     "Other": "The quote is not about protectionism or free trade"
            #     },
            "template_quote":
                {"Protectionism: Positive": "The quote is positive towards protectionism, for example protection of internal markets through tariffs or subsidies",
                 "Protectionism: Negative": "The quote is negative towards protectionism, for example in favour of free trade or open markets",
                 "Other": "The quote is not about protectionism or free trade"  # , free trade, tariffs
                 },
            "template_quote_2":
                {"Protectionism: Positive": "The quote is positive towards protectionism, for example in favour of protection of internal markets through tariffs or export subsidies or quotas",
                 "Protectionism: Negative": "The quote is negative towards protectionism, for example in favour of free trade or open international markets",
                 "Other": "The quote is not about protectionism or free trade"  # , free trade, tariffs
                 },
            "template_quote_context":
                {"Protectionism: Positive": "The quote is positive towards protectionism, for example protection of internal markets through tariffs or subsidies",
                 "Protectionism: Negative": "The quote is negative towards protectionism, for example in favour of free trade or open markets",
                 "Other": "The quote is not about protectionism or free trade"  # , free trade, tariffs
                 },
            "template_quote_2_context":
                {"Protectionism: Positive": "The quote is positive towards protectionism, for example in favour of protection of internal markets through tariffs or export subsidies or quotas",
                 "Protectionism: Negative": "The quote is negative towards protectionism, for example in favour of free trade or open international markets",
                 "Other": "The quote is not about protectionism or free trade"  # , free trade, tariffs
                 },
            "template_not_nli":
                {"placeholder": "placeholder"},
            "template_not_nli_context":
                {"placeholder": "placeholder"},
        })
        ## sort hypotheses alphabetically by label text
        # hypothesis_hyperparams_dic = OrderedDict()
        for key, value in hypothesis_hyperparams_dic.items():
            hypothesis_hyperparams_dic.update({key: dict(sorted(value.items(), key=lambda x: x[0]))})  # .lower()
        print(hypothesis_hyperparams_dic)
        print(dataset_name)

        ### text formatting function
        def format_text(df=None, text_format=None, embeddings=embeddings):
            if text_format == 'template_not_nli':
                df["text_prepared"] = df.text_original
            elif (text_format == 'template_not_nli_context') and (embeddings == False):
                df["text_prepared"] = df.text_preceding.fillna("") + " " + df.text_original.fillna("") + " " + df.text_following.fillna("")
            elif (text_format == 'template_not_nli_context') and (embeddings == True):  # for classical_ml with embedding input, including embeddings of context sentences
                df["text_prepared"] = df.apply(lambda x: np.mean([x["text_original"], x["text_original"], x["text_original"], x["text_original"], x["text_preceding"], x["text_following"]], dtype=object, axis=0), axis=1)  # weigh target text x times higher
            elif (text_format == 'template_quote') or (text_format == 'template_quote_2'):
                df["text_prepared"] = 'The quote: "' + df.text_original.fillna("") + '" - end of the quote.'
            elif (text_format == 'template_quote_context') or (text_format == 'template_quote_2_context'):
                df["text_prepared"] = df.text_preceding.fillna("") + '. The quote: "' + df.text_original.fillna("") + '" - end of the quote. ' + df.text_following.fillna("")
            # elif text_format == 'template_complex':
            #  df["text_prepared"] = df.text_preceding.fillna("") + df.text_original.fillna("") + df.text_following.fillna("")
            else:
                raise Exception(f'Hypothesis template not found for: {text_format}')
            return df.copy(deep=True)



    ### manifesto-morality
    if dataset_name == "manifesto-morality":
        hypothesis_hyperparams_dic = OrderedDict({
            "template_quote":
                {"Traditional Morality: Positive": "The quote is positive towards traditional morality",
                 "Traditional Morality: Negative": "The quote is negative towards traditional morality",
                 "Other": "The quote is not about traditional morality"
                 },
            "template_quote_2":
                {
                    "Traditional Morality: Positive": "The quote is positive towards traditional morality, for example in favour of traditional family values, religious institutions, or against unseemly behaviour",
                    "Traditional Morality: Negative": "The quote is negative towards traditional morality, for example in favour of divorce or abortion, modern families, separation of church and state, modern values",
                    "Other": "The quote is not about traditional morality, for example not about family values, abortion or religion"
                    },
            "template_quote_context":
                {"Traditional Morality: Positive": "The quote is positive towards traditional morality",
                 "Traditional Morality: Negative": "The quote is negative towards traditional morality",
                 "Other": "The quote is not about traditional morality"
                 },
            "template_quote_2_context":
                {
                    "Traditional Morality: Positive": "The quote is positive towards traditional morality, for example in favour of traditional family values, religious institutions, or against unseemly behaviour",
                    "Traditional Morality: Negative": "The quote is negative towards traditional morality, for example in favour of divorce or abortion, modern families, separation of church and state, modern values",
                    "Other": "The quote is not about traditional morality, for example not about family values, abortion or religion"
                },
            "template_not_nli":
                {"placeholder": "placeholder"},
            "template_not_nli_context":
                {"placeholder": "placeholder"},
        })
        ## sort hypotheses alphabetically by label text
        # hypothesis_hyperparams_dic = OrderedDict()
        for key, value in hypothesis_hyperparams_dic.items():
            hypothesis_hyperparams_dic.update({key: dict(sorted(value.items(), key=lambda x: x[0]))})  # .lower()
        print(hypothesis_hyperparams_dic)
        print(dataset_name)

        ### text formatting function
        def format_text(df=None, text_format=None, embeddings=embeddings):
            if text_format == 'template_not_nli':
                df["text_prepared"] = df.text_original
            elif (text_format == 'template_not_nli_context') and (embeddings == False):
                df["text_prepared"] = df.text_preceding.fillna("") + " " + df.text_original.fillna("") + " " + df.text_following.fillna("")
            elif (text_format == 'template_not_nli_context') and (embeddings == True):  # for classical_ml with embedding input, including embeddings of context sentences
                df["text_prepared"] = df.apply(lambda x: np.mean([x["text_original"], x["text_original"], x["text_original"], x["text_original"], x["text_preceding"], x["text_following"]], dtype=object, axis=0), axis=1)  # weigh target text x times higher
            elif (text_format == 'template_quote') or (text_format == 'template_quote_2'):
                df["text_prepared"] = 'The quote: "' + df.text_original.fillna("") + '" - end of the quote. '
            elif (text_format == 'template_quote_context') or (text_format == 'template_quote_2_context'):
                df["text_prepared"] = df.text_preceding.fillna("") + '. The quote: "' + df.text_original.fillna("") + '" - end of the quote. ' + df.text_following.fillna("")
            # elif text_format == 'template_complex':
            #  df["text_prepared"] = df.text_preceding.fillna("") + df.text_original.fillna("") + df.text_following.fillna("")
            else:
                raise Exception(f'Hypothesis template not found for: {text_format}')
            return df.copy(deep=True)


    ## return hypothesis_hyperparms_dic and function for corresponding text formatting
    print("Returning hypothesis hyperparameters dictionary and 'format_text' function for formatting text for hypothesis.")
    return hypothesis_hyperparams_dic, format_text


