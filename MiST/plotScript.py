import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np


# dictionary for colors
def GetPlotColor(cls):
    color_dict = {
         "TT_1b":   ROOT.kCyan-7,
         "TT_2b":   ROOT.kBlue,
         "TT_bb":   ROOT.kMagenta-4,
         "TT_cc":   ROOT.kGreen-1,
         "TT_lf":   ROOT.kGreen,
           "THQ":   ROOT.kRed,
           "THW":   ROOT.kPink,
           "TTH":   ROOT.kYellow-9,
        }
    return color_dict[cls]


def setupHistogram(
        values, nbins, bin_range,
        xtitle, ytitle, 
        color = ROOT.kBlack, filled = True):
    # define histogram
    histogram = ROOT.TH1F(xtitle, "", nbins, *bin_range)
    for v in zip(values):
      histogram.Fill(v)
    histogram.SetStats(False)
    histogram.GetXaxis().SetTitle(xtitle)
    histogram.GetYaxis().SetTitle(ytitle)

    histogram.GetYaxis().SetTitleOffset(1.4)
    histogram.GetXaxis().SetTitleOffset(1.2)
    histogram.GetYaxis().SetTitleSize(0.055)
    histogram.GetXaxis().SetTitleSize(0.055)
    histogram.GetYaxis().SetLabelSize(0.055)
    histogram.GetXaxis().SetLabelSize(0.055)

    histogram.SetMarkerColor(color)

    if filled:
       histogram.SetLineColor( ROOT.kBlack )
       histogram.SetFillColor( color )
       histogram.SetLineWidth(1)
    else:
       histogram.SetLineColor( color )
       histogram.SetFillColor(0)
       histogram.SetLineWidth(2)

    return histogram

def getLegend():
    legend=ROOT.TLegend(0.70,0.6,0.95,0.9)
    legend.SetBorderSize(0);
    legend.SetLineStyle(0);
    legend.SetTextFont(42);
    legend.SetTextSize(0.05);
    legend.SetFillStyle(0);
    return legend
 
def saveCanvas(canvas, path):
    canvas.SaveAs(path)
    canvas.SaveAs(path.replace(".pdf",".png"))
    canvas.Clear()

def getCanvas(name, ratiopad = False):
    if ratiopad:
        canvas = ROOT.TCanvas(name, name, 1024, 1024)
        canvas.Divide(1,2)
        canvas.cd(1).SetPad(0.,0.3,1.0,1.0)
        canvas.cd(1).SetTopMargin(0.07)
        canvas.cd(1).SetBottomMargin(0.0)

        canvas.cd(2).SetPad(0.,0.0,1.0,0.3)
        canvas.cd(2).SetTopMargin(0.0)
        canvas.cd(2).SetBottomMargin(0.4)

        canvas.cd(1).SetRightMargin(0.05)
        canvas.cd(1).SetLeftMargin(0.15)
        canvas.cd(1).SetTicks(1,1)

        canvas.cd(2).SetRightMargin(0.05)
        canvas.cd(2).SetLeftMargin(0.15)
        canvas.cd(2).SetTicks(1,1)
    else:
        canvas = ROOT.TCanvas(name, name, 1024, 768)
        canvas.SetTopMargin(0.07)
        canvas.SetBottomMargin(0.15)
        canvas.SetRightMargin(0.05)
        canvas.SetLeftMargin(0.15)
        canvas.SetTicks(1,1)
        return canvas
