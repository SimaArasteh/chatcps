import datasets
import glob
import os

class FunctionDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="@Cma will add description later",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    # "title": datasets.Value("string"),
                    "module": datasets.Value("string"),
                    "context": datasets.Value("string"), #TODO(@CMA: this is the amount we can fit in the prompt"
                    "function_name": datasets.Value("string"),
                    "function_body": datasets.Value("string"),
                    "label":datasets.Value("int8"),
                }
            ),
            # No default supervised_keys (as we have to pass both question --> check this later
            # and context as input).
            supervised_keys=None,
            homepage="", #TODO(@CMA add the homepage to the dataset"
            citation="", #TODO(@CMA/@JPegah add the citation to dataset"
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": 'cma_data/train/'}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": 'cma_data/test/'}),
        ]


    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""

        module_folders = [f.path for f in os.scandir(filepath) if f.is_dir()]
        for module in module_folders:
            module_name_index = module.rfind('/')
            module_name = module[module_name_index + 1:]
            for f in glob.glob(module + '/*.txt'):
                func_name_index = f.rfind('/')
                func_name = f[func_name_index + 1:-4]
                func_value = open(f).read()
                id_ = '-'.join([module_name, func_name])
                yield id_, {
                    "module": module_name,
                    "context": '', #TODO(@CMA what is a good amount of context for prompts?)
                    "function_name": func_name,
                    "function_body": func_value,
                    "id": id_,
                    "label": 0

                }