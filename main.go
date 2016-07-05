package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"

	"github.com/mjibson/go-dsp/fft"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/num-analysis/linalg/ludecomp"
	"github.com/unixpickle/num-analysis/linalg/qrdecomp"
	"github.com/unixpickle/wav"
)

const (
	WindowSize   = 512
	MinAmplitude = 1e-2
	Damping      = 1e-5
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

	if len(sourceVecs) < WindowSize {
		fmt.Fprintf(os.Stderr, "Not enough samples (have %d need %d)\n",
			len(sourceVecs), WindowSize)
		os.Exit(1)
	}

	sourceMat := vecMatrix(sourceVecs)
	targetMat := vecMatrix(targetVecs)

	log.Printf("Creating QR decomposition with %d samples...", len(sourceVecs))
	for i := 0; i < sourceMat.Cols && i < sourceMat.Rows; i++ {
		sourceMat.Set(i, i, sourceMat.Get(i, i)+Damping)
	}
	q, r := qrdecomp.Householder(sourceMat)
	rInv := ludecomp.Decompose(r)
	qInv := q.Transpose()

	log.Println("Solving least-squares matrix...")
	bTranspose := qInv.Mul(targetMat).Transpose()
	var result linalg.Vector
	for i := 0; i < bTranspose.Rows; i++ {
		rowVec := bTranspose.Data[bTranspose.Cols*i : bTranspose.Cols*(i+1)]
		result = append(result, rInv.Solve(rowVec)...)
	}

	log.Println("Saving result...")
	resultMat := &linalg.Matrix{
		Rows: WindowSize,
		Cols: WindowSize,
		Data: result,
	}
	encoded, _ := json.Marshal(resultMat)
	if err := ioutil.WriteFile(outPath, encoded, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save:", err)
		os.Exit(1)
	}

	log.Println("Measuring error...")
	var targetMag, totalError, totalTransError float64
	for i, sourceVec := range sourceVecs {
		negTarget := targetVecs[i].Copy().Scale(-1)

		targetMag += negTarget.Dot(negTarget)

		diff := sourceVec.Copy().Add(negTarget)
		totalError += diff.Dot(diff)

		newSource := linalg.Vector(resultMat.Mul(linalg.NewMatrixColumn(sourceVec)).Data)
		diff1 := newSource.Add(negTarget)
		totalTransError += diff1.Dot(diff1)
	}
	log.Println("Total error:", totalError, "(no trans)", totalTransError, "(trans)",
		targetMag, "(target mag)")
}

func translateCommand() {
	if len(os.Args) != 5 {
		dieUsage()
	}

	outPath := os.Args[4]

	matData, err := ioutil.ReadFile(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read matrix:", err)
		os.Exit(1)
	}

	var mat linalg.Matrix
	if err := json.Unmarshal(matData, &mat); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to parse matrix:", err)
		os.Exit(1)
	}

	source, err := wav.ReadSoundFile(os.Args[3])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read source:", err)
		os.Exit(1)
	}

	inSamples := source.Samples()
	var outSamples []wav.Sample

	inVec := linalg.NewMatrix(mat.Rows, 1)
	for i := 0; i+mat.Rows <= len(inSamples); i += mat.Rows {
		for j := 0; j < mat.Rows; j++ {
			inVec.Data[j] = float64(inSamples[i+j])
		}
		forwardFFT(inVec.Data)
		outVec := mat.Mul(inVec).Data
		backwardFFT(outVec)
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
		forwardFFT(inVec)
		forwardFFT(outVec)
		if inVec.Dot(inVec) < MinAmplitude || outVec.Dot(outVec) < MinAmplitude {
			continue
		}
		inVecs = append(inVecs, inVec)
		outVecs = append(outVecs, outVec)
	}

	return
}

func vecMatrix(rows []linalg.Vector) *linalg.Matrix {
	colCount := len(rows[0])
	res := &linalg.Matrix{
		Rows: len(rows),
		Cols: colCount,
		Data: make(linalg.Vector, colCount*len(rows)),
	}
	var idx int
	for _, x := range rows {
		copy(res.Data[idx:], x)
		idx += len(x)
	}
	return res
}

func forwardFFT(data linalg.Vector) {
	res := fft.FFTReal(data)
	for i := 0; i < len(res); i++ {
		res[i] = complex(real(res[i]), 0)
	}
	/*for i := 200; i < len(res); i++ {
		res[i] = 0
	}*/
	for i, x := range res {
		data[i] = real(x)
	}
}

func backwardFFT(data linalg.Vector) {
	res := make([]complex128, len(data))
	for i, x := range data {
		res[i] = complex(x, 0)
	}
	res = fft.IFFT(res)
	for i, x := range res {
		data[i] = real(x)
	}
}
