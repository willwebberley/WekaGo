// Copyright 2015 Will Webberley
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package wekago

import (
        "os/exec"
        "os"
        "io/ioutil"
        "strings"
        "strconv"
        "errors"
        "math/rand"
)

/* 
    Declare required structs
*/
type prediction struct{
    Index int64
    Observed_value string
    Predicted_value string
    Probability float64
}

type instance struct{
    features []feature
}

type feature struct{
    name string
    value string
    possible_values string
}

type model struct{
    classifier string
    training_file string
    testing_file string
    model_file string
    training_instances []instance
    testing_instances []instance
    model_dir string
    arff_dir string
    Predictions []prediction
    id string
    trained bool
}


/*
    Helper functions to initialise new types
*/
func NewModel(classifier string) model{
    model := model{classifier:classifier, id:"test"}
    model.model_dir = "wekago_data/models"
    model.arff_dir = "wekago_data/arff"
    model.id = strconv.Itoa(rand.Int())
    model.trained = false
    os.MkdirAll(model.model_dir, 0777)
    os.MkdirAll(model.arff_dir, 0777)

    return model
}

func NewFeature(name, value, possible_values string) feature{
    feature := feature{}
    feature.name = name
    feature.value = value
    feature.possible_values = possible_values
    return feature
}

func NewInstance() instance{
    instance := instance{}
    return instance
}


/*
    Methods for model struct
*/
func (m *model) AddTrainingInstance(instance instance){
    m.training_instances = append(m.training_instances, instance)
}

func (m *model) AddTestingInstance(instance instance){
    m.testing_instances = append(m.testing_instances, instance)
}

func (m *model) createARFF(file_type string, instances []instance) (error){
    file_name := m.arff_dir+"/"+m.id+"_"+file_type+".arff"
    output := []byte("@relation "+m.id+"\n")
    for index, instance := range instances{
        if index == 0{
            for _, feature := range instance.features{
                output = append(output, "\t@attribute "+feature.name+" "+feature.possible_values+"\n"...)
            }
            output = append(output, "\n@data\n"...)
        }
        for index2, feature := range instance.features{
            if index2 == 0{
                output = append(output, feature.value...)
            }else{
                output = append(output, ","+feature.value...)
            }
        }
        output = append(output," \n"...)
    }
    err := ioutil.WriteFile(file_name, output, 0644)
    if err != nil{
        return errors.New("Couldn't write ARFF file.")
    }
    if file_type == "training"{
        m.training_file = file_name
    }
    if file_type == "testing"{
        m.testing_file = file_name
    }
    return nil
}

func (m *model) Train() error{
    write_err := m.createARFF("training", m.training_instances)
    if write_err != nil{
        return write_err
    }
    save_as := m.model_dir+"/"+m.id+".model"
    output, java_err := exec.Command("java", "-cp", "./weka.jar", "weka.classifiers."+m.classifier, "-t", m.training_file, "-d", save_as).Output()
    if java_err != nil || len(output) == 0{
        return errors.New("Error invoking JVM. Ensure your CLASSPATH is properly set and that your classifier is valid.")
    }
    m.trained = true
    m.model_file = save_as
    return nil
}

func (m *model) LoadModel(model_file string){
    m.model_file = model_file
    m.trained = true
}

func (m *model) Test() error{
    if m.model_file == "" || !m.trained{
        return errors.New("Need to train model first.")
    }

    m.createARFF("testing", m.testing_instances)
    output, err := exec.Command("java", "-cp", "./weka.jar", "weka.classifiers."+m.classifier, "-T", m.testing_file, "-l", m.model_file, "-p", "0").Output()
    if err != nil || len(output) == 0{
        return errors.New("Error invoking JVM. Ensure your CLASSPATH is properly set and that your classifier is valid.")
    }

    out_string := string(output[:len(output)])
    lines := strings.Split(out_string, "\n") 
    for _, line := range lines{
        data := strings.Fields(line)
        if len(data) >= 4 && data[0] != "===" && data[0] != "inst#"{
            p := prediction{}
            p.Index, _ = strconv.ParseInt(data[0], 0, 64)
            p.Observed_value = strings.Split(data[1], ":")[1]
            p.Predicted_value = strings.Split(data[2], ":")[1]
            p.Probability, _ = strconv.ParseFloat(data[3], 64)
            m.Predictions = append(m.Predictions, p)
        }
    }
    return nil;
}


/* 
    Methods for instance struct
*/
func (i *instance) AddFeature(feature feature){
    i.features = append(i.features, feature)
}
