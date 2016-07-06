package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"os"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/hessfree"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/wav"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	WindowSize   = 1024
	MinAmplitude = 1e-2
	HiddenSize1  = 300
	HiddenSize2  = 500
	MaxSubBatch  = 20
)

func main() {
	if len(os.Args) < 2 {
		dieUsage()
	}
	switch os.Args[1] {
	case "gen":
		genCommand()
	case "translate":
		translateCommand()
	default:
		dieUsage()
	}
}

func dieUsage() {
	fmt.Fprintln(os.Stderr, "Usage: voicechange gen <source.wav> <target.wav> <output.json>\n"+
		"                   translate <mat.json> <input.wav> <output.wav>")
	os.Exit(1)
}

func genCommand() {
	if len(os.Args) != 5 {
		dieUsage()
	}

	outPath := os.Args[4]
	sourceVecs, targetVecs := sourceTargetVecs(readAudioFiles(os.Args[2], os.Args[3]))
	samples := neuralnet.VectorSampleSet(sourceVecs, targetVecs)

	network := neuralnet.Network{
		/*&neuralnet.RescaleLayer{
			Bias:  -average,
			Scale: 1 / stddev,
		},*/
		&neuralnet.DenseLayer{
			InputCount:  WindowSize,
			OutputCount: HiddenSize1,
		},
		neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize1,
			OutputCount: HiddenSize2,
		},
		neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize2,
			OutputCount: WindowSize,
		},
	}
	network.Randomize()

	ui := hessfree.NewConsoleUI()
	learner := &hessfree.DampingLearner{
		WrappedLearner: &hessfree.NeuralNetLearner{
			Layers:         network,
			Output:         nil,
			Cost:           neuralnet.MeanSquaredCost{},
			MaxSubBatch:    MaxSubBatch,
			MaxConcurrency: 2,
		},
		DampingCoeff: 0.1,
		UseQuadMin:   true,
		UI:           ui,
	}
	trainer := hessfree.Trainer{
		Learner:   learner,
		Samples:   samples,
		BatchSize: samples.Len(),
		UI:        ui,
		Convergence: hessfree.ConvergenceCriteria{
			MinK: 5,
		},
	}
	trainer.Train()

	encoded, _ := network.Serialize()
	if err := ioutil.WriteFile(outPath, encoded, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save:", err)
		os.Exit(1)
	}
}

func translateCommand() {
	if len(os.Args) != 5 {
		dieUsage()
	}

	outPath := os.Args[4]

	netData, err := ioutil.ReadFile(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read network:", err)
		os.Exit(1)
	}

	network, err := neuralnet.DeserializeNetwork(netData)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to parse network:", err)
		os.Exit(1)
	}

	source, err := wav.ReadSoundFile(os.Args[3])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read source:", err)
		os.Exit(1)
	}

	inSamples := source.Samples()
	var outSamples []wav.Sample

	inVec := make(linalg.Vector, WindowSize)
	for i := 0; i+WindowSize <= len(inSamples); i += WindowSize {
		for j := 0; j < WindowSize; j++ {
			inVec[j] = float64(inSamples[i+j])
		}
		outVec := network.Apply(&autofunc.Variable{Vector: inVec}).Output()
		for _, k := range outVec {
			k = math.Max(math.Min(k, 1), -1)
			outSamples = append(outSamples, wav.Sample(k))
		}
	}

	source.SetSamples(outSamples)
	if err := wav.WriteFile(source, outPath); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to write result:", err)
		os.Exit(1)
	}
}

func readAudioFiles(sourcePath, targetPath string) (source, target wav.Sound) {
	source, err := wav.ReadSoundFile(sourcePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read source:", err)
		os.Exit(1)
	}

	target, err = wav.ReadSoundFile(targetPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read target:", err)
		os.Exit(1)
	}

	if len(source.Samples()) != len(target.Samples()) {
		fmt.Fprintln(os.Stderr, "Source and target must have matching sizes.")
		os.Exit(1)
	}

	return
}

func sourceTargetVecs(source, target wav.Sound) (inVecs, outVecs []linalg.Vector) {
	inSamples := source.Samples()
	outSamples := target.Samples()

	for i := 0; i+WindowSize <= len(inSamples); i += WindowSize {
		inVec := make(linalg.Vector, WindowSize)
		outVec := make(linalg.Vector, WindowSize)
		for j := 0; j < WindowSize; j++ {
			inVec[j] = float64(inSamples[i+j])
			outVec[j] = float64(outSamples[i+j])
		}
		if inVec.Dot(inVec) < MinAmplitude || outVec.Dot(outVec) < MinAmplitude {
			continue
		}
		inVecs = append(inVecs, inVec)
		outVecs = append(outVecs, outVec)
	}

	return
}
