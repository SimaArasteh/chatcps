import datasets
import glob
import os
import numpy as np

class ModuleDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="@Cma will add description later",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    # "title": datasets.Value("string"),
                    "module": datasets.Value("string"),
                    "context": datasets.Value("string"), #TODO(@CMA: this is the amount we can fit in the prompt"
                    # "function_name": datasets.Value("string"),
                    # "function_body": datasets.Value("string"),
                    "knowledge":datasets.Value("string"),
                    "label":datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "options":datasets.Value("string"),
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

    
    # the criteria for choosing modules
    def top_func_modules(self, module_folder):
        funcs = []
        funcs_size = []
        
        # create the function size mappings
        for f in glob.glob(module_folder + '/*.txt'):
            func_name_index = f.rfind('/')
            func_name = f[func_name_index + 1:-4]
            func_value = open(f).read()
            tmp = func_value.split('\n')
            funcs.append(func_value)
            funcs_size.append(len(tmp))
        
        np_size=np.array(funcs_size)
        
        # currently I am returning top 20 functions
        top_inds = np.argsort(np_size)[:20]
        print('This is top functions indexes and their sizes', top_inds, '| ', np_size[:top_inds])
        
        funcs_joined = '\n'.join([funcs[i] for i in top_inds])
        return funcs_joined
      

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""       
        knowledge="""Here is the definition of module categories category in a softwared system. 
            Navigation module: contains functions that calculate the position of the vehicle \n
            Communication module: contains functions that implement any communication protocols such as uart, spi or any sign of packet or message sending.\n
            File system management: contains functions that read or write from a file or similar tasks.\n
            Controller: contains functions that control the vehicle based on the position and parameters.\n
            Safety check: contains functions that check if the activity is safe to do.\n
            """
        prompt_instruction="""II will give you a module that contains decompiled functions from a control system firmware such as copter. Tell me what is the category of this module.decide based on the majority of functions."""
        
        options=["Navigation", "Communication", "File system management", "Controller", "Safety check"]
            
            
        module_folders = [f.path for f in os.scandir(filepath) if f.is_dir()]
        for module in module_folders:
            module_name_index = module.rfind('/')
            module_name = module[module_name_index + 1:]

            # build module context based on the given criteria

            id_ = module_name
            selected_context = self.top_func_modules(module)
            yield id_, {
                "module": module_name,
                "context": selected_context, #TODO(@CMA what is a good amount of context for prompts?)
                "knowledge": "",
                "instruction": prompt_instruction,
                "options":options,
                "id": id_,
                "label": "Navigation"

            }
