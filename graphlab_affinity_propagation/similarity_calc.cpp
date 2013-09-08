#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <limits>
#include <vector>
#include <algorithm>
#include <graphlab.hpp>
using namespace std;

struct vertex_data {
	std::vector<double> feature;	
	std::vector<double> sims;

	void save(graphlab::oarchive& oarc) const {
		oarc << feature << sims;
	}

	void load(graphlab::iarchive& iarc) {
		iarc >> feature >> sims;
	}
};

double eucliean(const std::vector<double>& a, const std::vector<double>& b) {
	ASSERT_EQ(a.size(), b.size());

	double sum = 0.0;
	for(size_t i = 0; i < a.size(); i++){	
		sum += (a[i] - b[i]) * (a[i] - b[i]);	
	}

	return -sqrt(sum);
}

typedef graphlab::distributed_graph<vertex_data, graphlab::empty> graph_type;

graphlab::atomic<graphlab::vertex_id_type> NEXT_VID;

bool vertex_loader(graph_type& graph, const std::string& fname, const std::string& line) {

	if (line.empty())
		return true;

	vertex_data vtx;

	using boost::lexical_cast;
	using boost::bad_lexical_cast;

	graphlab::vertex_id_type vid;

	std::string str = line;
	std::string delimiter = ",";
	size_t pos = 0;

	int cnt = 0;
	while((pos = str.find(delimiter)) != std::string::npos) {
		try {
			if (cnt == 0)
				vid = lexical_cast<graphlab::vertex_id_type>(str.substr(0, pos));
			
			else
				vtx.feature.push_back(lexical_cast<double>(str.substr(0, pos)));
			cnt++;
		} catch(bad_lexical_cast &) {
			return false;
		}
		str.erase(0, pos+delimiter.length());
	}	

	delimiter = "\t";
	pos = 0;	

	graph.add_vertex(vid, vtx);		

	while((pos = str.find(delimiter)) != std::string::npos) {
		try {
			graph.add_edge(vid, lexical_cast<int>(str.substr(0,pos)), graphlab::empty());							
		} catch(bad_lexical_cast &) {
			return false;
		}
		str.erase(0, pos + delimiter.length());
	}	

	try {
		graph.add_edge(vid, lexical_cast<int>(str), graphlab::empty());
	} catch (bad_lexical_cast &) {
		return false;
	}
	
	return true;
}

struct set_union_gather : public graphlab::IS_POD_TYPE {	

	set_union_gather() {}

	explicit set_union_gather(const graph_type::vertex_type& a_source, const graph_type::edge_type& a_target){	
	
		double sim = eucliean(a_source.data().feature, a_target.source().data().feature);
		graph_type::vertex_type source = a_source;
		source.data().sims.push_back(sim);
	}
	
	set_union_gather& operator+=(const set_union_gather& other) {		
		return *this;
	}	
};

struct vertex_writer {

	std::string save_vertex(graph_type::vertex_type v) {
		std::stringstream strm;

		bool p = false;
		for(size_t i = 0; i < v.data().sims.size();i++) {
			if (i == v.id() && !p) {
				strm << 0.0 << "\t";
				i--;
				p = true;
			}
			else
				strm << v.data().sims[i] << "\t";
		}
		if (p == false)
			strm << 0.0 << "\t";
		strm << "\n";

		strm.flush();

		return strm.str();
	}

	std::string save_edge(graph_type::edge_type e) {return "";}
};


class Similarity_Calc : public graphlab::ivertex_program<graph_type, set_union_gather>, public graphlab::IS_POD_TYPE {
	public:


		edge_dir_type gather_edges(icontext_type& context,
				const vertex_type& vertex) const {
			return graphlab::IN_EDGES;
		}

		gather_type gather(icontext_type& context, const vertex_type& vertex, edge_type& edge) const {		
			return set_union_gather(vertex, edge);
		}

		void apply(icontext_type& context, vertex_type& vertex,
				const gather_type& total) {	
		}

		edge_dir_type scatter_edges(icontext_type& context,
				const vertex_type& vertex) const {
			//return graphlab::OUT_EDGES;
		}

		void scatter(icontext_type& context, const vertex_type& vertex,
				edge_type& edge) const {
		}

};

int main(int argc, char** argv) {


	graphlab::command_line_options clopts
		("K-means clustering. The input data file is provided by the "
		 "--data argument which is non-optional. The format of the data file is a "
		 "collection of lines, where each line contains a comma or white-space "
		 "separated lost of numeric values representing a vector. Every line "
		 "must have the same number of values. The required --clusters=N "
		 "argument denotes the number of clusters to generate. To store the output "
		 "see the --output-cluster and --output-data arguments");

	std::string datafile;
	std::string outcluster_file;
	std::string outdata_file;
	clopts.attach_option("data", datafile,
			"Input file. Each line hold a white-space or comma separated numeric vector");	
	clopts.attach_option("output-clusters", outcluster_file,
			"If set, will write a file containing cluster centers "
			"to this filename. This must be on the local filesystem "
			"and must be accessible to the root node.");
	clopts.attach_option("output-data", outdata_file,
			"If set, will output a copy of the input data with an additional "
			"last column denoting the assigned cluster centers. The output "
			"will be written to a sequence of filenames where each file is "
			"prefixed by this value. This may be on HDFS.");


	if(!clopts.parse(argc, argv)) return EXIT_FAILURE;
	if (datafile == "") {
		std::cout << "--data is not optional\n";
		clopts.print_description();
		return EXIT_FAILURE;
	}

	graphlab::mpi_tools::init(argc, argv);
	graphlab::distributed_control dc;


	// load graph
	graph_type graph(dc, clopts);
	NEXT_VID = dc.procid();
	graph.load(datafile, vertex_loader);

	graph.finalize();

	typedef graphlab::omni_engine<Similarity_Calc> engine_type;

	engine_type engine(dc, graph, "sync", clopts);
	engine.signal_all();
	graphlab::timer timer;
	engine.start();

	//写数据点
	if (!outdata_file.empty()) {
		dc.cout() << "Writing Data and its vesting cluster to file " << outdata_file << std::endl;
		graph.save(outdata_file, vertex_writer(), false, true, false, 1);
	}

	graphlab::mpi_tools::finalize();

}	
